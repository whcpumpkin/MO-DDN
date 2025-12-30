from thirdparty.GLEE.glee.models.glee_model import GLEE_Model
from thirdparty.GLEE.glee.config_deeplab import add_deeplab_config
from thirdparty.GLEE.glee.config import add_glee_config
# from habitat_sim.utils.common import d3_40_colors_rgb
from constants import *
from detectron2.config import get_cfg
from .object_list import categories as CATEGORIES
import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np

CATEGORIES = [cat['name'].lower() for cat in CATEGORIES]
d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)


def initialize_glee(glee_config=GLEE_CONFIG_PATH, glee_checkpoint=GLEE_CHECKPOINT_PATH, device="cuda:0"):
    cfg_swin = get_cfg()
    add_deeplab_config(cfg_swin)
    add_glee_config(cfg_swin)
    conf_files_swin = glee_config
    checkpoints_swin = torch.load(glee_checkpoint)
    cfg_swin.merge_from_file(conf_files_swin)
    GLEEmodel_swin = GLEE_Model(cfg_swin, None, device, None, True).to(device)
    GLEEmodel_swin.load_state_dict(checkpoints_swin, strict=False)
    GLEEmodel_swin.eval()
    return GLEEmodel_swin


def glee_segmentation(img, GLEEmodel, prompt_mode="categories", custom_category=CATEGORIES, results_select=["box", "mask", "name", "score"], num_inst_select=15, threshold_select=0.2, device="cuda:0"):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).to(device).view(3, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    ori_image = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
    ori_image = normalizer(ori_image.to(device))[
        None,
    ]
    _, _, ori_height, ori_width = ori_image.shape
    resizer = torchvision.transforms.Resize(800, antialias=True)
    resize_image = resizer(ori_image)
    image_size = torch.as_tensor((resize_image.shape[-2], resize_image.shape[-1]))
    re_size = resize_image.shape[-2:]
    stride = 32
    # the last two dims are H,W, both subject to divisibility requirement
    padding_size = ((image_size + (stride - 1)).div(stride, rounding_mode="floor") * stride).tolist()
    infer_image = torch.zeros(1, 3, padding_size[0], padding_size[1]).to(resize_image)
    infer_image[0, :, :image_size[0], :image_size[1]] = resize_image
    batch_category_name = custom_category
    prompt_list = []
    with torch.no_grad():
        (outputs, _) = GLEEmodel(infer_image, prompt_list, task="coco", batch_name_list=batch_category_name, is_train=False)
    topK_instance = max(num_inst_select, 1)
    mask_pred = outputs['pred_masks'][0]
    mask_cls = outputs['pred_logits'][0]
    scores = mask_cls.sigmoid().max(-1)[0]
    scores_per_image, topk_indices = scores.topk(topK_instance, sorted=True)
    valid = scores_per_image > threshold_select
    topk_indices = topk_indices[valid]
    scores_per_image = scores_per_image[valid]
    pred_class = mask_cls[topk_indices].max(-1)[1].tolist()
    if len(pred_class) == 0:
        return [], [], []
    mask_pred = mask_pred[topk_indices]
    pred_masks = F.interpolate(mask_pred[
        None,
    ], size=(padding_size[0], padding_size[1]), mode="bilinear", align_corners=False)
    pred_masks = pred_masks[:, :, :re_size[0], :re_size[1]]
    pred_masks = F.interpolate(pred_masks, size=(ori_height, ori_width), mode="bilinear", align_corners=False)
    pred_masks = (pred_masks > 0).detach().cpu().numpy()[0]
    return pred_masks, np.array(batch_category_name)[pred_class], scores_per_image


def visualize_segmentation(image, classes, masks):
    copy_image = image.copy()
    label_classes = np.unique(classes)
    for cls, mask in zip(classes, masks):
        if len(np.unique(mask)) != 2: continue
        copy_image[np.where(mask == 1)] = d3_40_colors_rgb[label_classes.tolist().index(cls)]
        x, y = int(np.mean(np.where(mask)[1])), int(np.mean(np.where(mask)[0]))
        cv2.putText(copy_image, str(cls), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    ret_image = cv2.addWeighted(image, 0.2, copy_image, 0.8, 0)
    return ret_image
