from constants import *
from .glee_detector import *
class GLEE_Percevior:
    def __init__(self,
                 glee_config=GLEE_CONFIG_PATH,
                 glee_checkpoint=GLEE_CHECKPOINT_PATH,
                 device = "cuda:0"):
        self.device = device
        self.glee_model = initialize_glee(glee_config,glee_checkpoint,device)
    def perceive(self,image,confidence_threshold=0.25,area_threshold=1600):
        pred_masks, pred_class, pred_confidence = glee_segmentation(image,self.glee_model,threshold_select=confidence_threshold,device=self.device)
        visualization = visualize_segmentation(image,pred_class,pred_masks)
        try:
            mask_area = np.array([mask.sum() for mask in pred_masks])
            return pred_class[mask_area>area_threshold],pred_masks[mask_area>area_threshold],pred_confidence[mask_area>area_threshold],[visualization]
        except:
            return [],[],[],[visualization]