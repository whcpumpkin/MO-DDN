from sklearn.cluster import KMeans
import numpy as np
import json
import transformers
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from attributebook.functions import vq, vq_st
from itertools import chain
import torch.nn.init as init
import torchvision.transforms as transforms


class VQEmbedding(nn.Module):

    def __init__(self, K, D, weight=None):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        if weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weight))
        else:
            self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        # z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x, indices = vq_st(z_e_x, self.embedding.weight.detach())
        # z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_q_x)
        # z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar_, indices


def make_mlp_layers(in_dim, out_dim, num_layers, hidden_dim):
    layers = []
    for i in range(num_layers - 1):
        if i == 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
        else:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class AttributeModel(nn.Module):

    def __init__(self, args):
        super(AttributeModel, self).__init__()
        self.args = args
        self.attribute_dim = args.transformer_embedding_dim
        self.CLIP_model, _ = clip.load("ViT-L/14", device=args.device)
        self.CLIP_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        if args.ins_encoder == 'clip':
            self.ins_encoder_model = self.CLIP_model
            self.ins_encoder_dim = self.attribute_dim
        elif args.ins_encoder == 'bgem3':
            self.ins_encoder_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            self.ins_encoder_dim = 1024

        self.instruction_MLP = make_mlp_layers(self.ins_encoder_dim, args.attribute_k1 * self.attribute_dim, self.args.num_of_MLP_layers, 2048)
        self.init_mlp_layers(self.instruction_MLP)
        self.object_MLP = make_mlp_layers(self.attribute_dim, args.attribute_k2 * self.attribute_dim, self.args.num_of_MLP_layers, 2048)
        self.init_mlp_layers(self.object_MLP)
        self.instruction_recon_decoder = make_mlp_layers(args.attribute_k1 * self.attribute_dim, self.ins_encoder_dim, self.args.num_of_MLP_layers, 2048)
        self.init_mlp_layers(self.instruction_recon_decoder)
        self.object_recon_decoder = make_mlp_layers(args.attribute_k2 * self.attribute_dim, self.attribute_dim, self.args.num_of_MLP_layers, 2048)
        self.init_mlp_layers(self.object_recon_decoder)

        self.load_initial_codebook()

    def init_mlp_layers(self, layers):
        for layer in layers:
            if isinstance(layer, nn.Linear):
                init.orthogonal_(layer.weight)
                init.zeros_(layer.bias)

    def load_initial_codebook(self):
        if "no_init_codebook" in self.args.path_to_init_codebook_file:
            self.codebook = VQEmbedding(codebook.shape[0], codebook.shape[1], None)
        else:
            codebook = np.load(self.args.path_to_init_codebook_file)
            self.codebook = VQEmbedding(codebook.shape[0], codebook.shape[1], codebook)

    def forward_instruction(self, instruction):
        if isinstance(instruction, str):
            instruction = [instruction]
        bs = len(instruction)
        if self.args.ins_encoder == 'bgem3':
            with torch.no_grad():
                flag_embedding = self.ins_encoder_model.encode(instruction)['dense_vecs']
            flag_embedding = torch.FloatTensor(flag_embedding).to(self.args.device)
        elif self.args.ins_encoder == 'clip':
            with torch.no_grad():
                flag_embedding = self.ins_encoder_model.encode_text(clip.tokenize(instruction).to(self.args.device))
                flag_embedding = flag_embedding.float()
        return flag_embedding

    def forward_object(self, object):
        if isinstance(object, str) or isinstance(object, list):
            text = clip.tokenize(object).to(self.args.device)
            with torch.no_grad():
                clip_feature = self.CLIP_model.encode_text(text)
        elif isinstance(object, np.ndarray) or isinstance(object, torch.Tensor):
            image = self.CLIP_preprocess(object).to(self.args.device)
            with torch.no_grad():
                clip_feature = self.CLIP_model.encode_image(image)
        clip_feature = clip_feature.to(torch.float64)
        clip_feature /= clip_feature.norm(dim=-1, keepdim=True)
        clip_feature = clip_feature.to(torch.float32)
        return clip_feature

    def forward_attribute(self, attribute):
        text = clip.tokenize(attribute).to(self.args.device)
        with torch.no_grad():
            clip_feature = self.CLIP_model.encode_text(text)
        clip_feature /= clip_feature.norm(dim=-1, keepdim=True)
        clip_feature = clip_feature.float()
        return clip_feature

    def forward(self, batch):
        ins = batch[0]
        ins_attribute = batch[1]
        obj_attribute = batch[2]
        bs = len(ins)
        ins_encode_feature = self.forward_instruction(ins)
        ins_encode_feature = ins_encode_feature / ins_encode_feature.norm(dim=-1, keepdim=True)
        ins_attribute_feature = self.instruction_MLP(ins_encode_feature).reshape(bs, self.args.attribute_k1, self.attribute_dim)
        ins_attribute_feature = ins_attribute_feature / ins_attribute_feature.norm(dim=-1, keepdim=True)

        obj_list = [[obj_att[0] for obj_att in objs] for objs in obj_attribute]
        obj_list = list(chain.from_iterable(obj_list))
        assert len(obj_list) == bs * self.args.attribute_k3
        obj_clip_feature = self.forward_object(obj_list).reshape(bs, self.args.attribute_k3, self.attribute_dim)
        obj_attribute_feature = self.object_MLP(obj_clip_feature.float()).reshape(bs, self.args.attribute_k3, self.args.attribute_k2, self.attribute_dim)
        obj_attribute_feature = obj_attribute_feature / obj_attribute_feature.norm(dim=-1, keepdim=True)

        ins_attribute_list = list(chain.from_iterable(ins_attribute))
        assert len(ins_attribute_list) == bs * self.args.attribute_k1
        ins_attribute_feature_gt = self.forward_attribute(ins_attribute_list).reshape(bs, self.args.attribute_k1, self.attribute_dim)

        obj_attribute_list = [[obj_att[1] for obj_att in objs] for objs in obj_attribute]
        obj_attribute_list = list(chain.from_iterable(obj_attribute_list))
        obj_attribute_list = list(chain.from_iterable(obj_attribute_list))
        assert len(obj_attribute_list) == bs * self.args.attribute_k3 * self.args.attribute_k2
        obj_attribute_feature_gt = self.forward_attribute(obj_attribute_list).reshape(bs, self.args.attribute_k3, self.args.attribute_k2, self.attribute_dim)

        ins_attribute_feature_q_st, ins_attribute_feature_q, idx_ins = self.codebook.straight_through(ins_attribute_feature)
        obj_attribute_feature_q_st, obj_attribute_feature_q, idx_obj = self.codebook.straight_through(obj_attribute_feature)

        recon_ins_encode_feature = self.instruction_recon_decoder(ins_attribute_feature_q_st.reshape(bs, self.args.attribute_k1 * self.attribute_dim))
        recon_obj_clip_feature = self.object_recon_decoder(obj_attribute_feature_q_st.reshape(bs, self.args.attribute_k3, self.args.attribute_k2 * self.attribute_dim))
        recon_ins_encode_feature = recon_ins_encode_feature / recon_ins_encode_feature.norm(dim=-1, keepdim=True)
        recon_obj_clip_feature = recon_obj_clip_feature / recon_obj_clip_feature.norm(dim=-1, keepdim=True)

        return ins_encode_feature, ins_attribute_feature, obj_clip_feature, obj_attribute_feature, ins_attribute_feature_gt, obj_attribute_feature_gt, recon_ins_encode_feature, recon_obj_clip_feature, ins_attribute_feature_q_st, obj_attribute_feature_q_st, ins_attribute_feature_q, obj_attribute_feature_q

    def forward_test(self, batch):
        ins = batch[0]
        basic_object = batch[1]
        preference_object = batch[2]
        bs = len(ins)
        ins_attribute_feature = None
        basic_object_attribute_feature = None
        preference_attribute_feature = None
        idx_ins = None
        idx_basic = None
        idx_preference = None

        ins_encode_feature = self.forward_instruction(ins)
        ins_encode_feature = ins_encode_feature / ins_encode_feature.norm(dim=-1, keepdim=True)
        ins_attribute_feature = self.instruction_MLP(ins_encode_feature).reshape(bs, self.args.attribute_k1, self.attribute_dim)
        ins_attribute_feature = ins_attribute_feature / ins_attribute_feature.norm(dim=-1, keepdim=True)
        ins_attribute_feature_q_st, ins_attribute_feature_q, idx_ins = self.codebook.straight_through(ins_attribute_feature)

        basic_object = list(chain.from_iterable(basic_object))
        basic_object = list(chain.from_iterable(basic_object))
        if len(basic_object) > 0:
            if any(isinstance(basic_object[i], list) for i in range(len(basic_object))):
                t = 1

        basic_object = [obj_name.split(".n")[0] for obj_name in basic_object]
        basic_object = list(set(basic_object))
        if len(basic_object) > 0:
            basic_object_clip_feature = self.forward_object(basic_object).reshape(bs, -1, self.attribute_dim)
            basic_object_attribute_feature = self.object_MLP(basic_object_clip_feature.float()).reshape(bs, -1, self.args.attribute_k2, self.attribute_dim)
            basic_object_attribute_feature = basic_object_attribute_feature / basic_object_attribute_feature.norm(dim=-1, keepdim=True)
            basic_object_attribute_feature_q_st, basic_object_attribute_feature_q, idx_basic = self.codebook.straight_through(basic_object_attribute_feature)

        preference_object = list(chain.from_iterable(preference_object))
        preference_object = list(chain.from_iterable(preference_object))
        if len(preference_object) > 0:
            if any(isinstance(preference_object[i], list) for i in range(len(preference_object))):
                t = 1
        preference_object = [obj_name.split(".n")[0] for obj_name in preference_object]
        preference_object = list(set(preference_object))
        if len(preference_object) > 0:
            preference_clip_feature = self.forward_object(preference_object).reshape(bs, -1, self.attribute_dim)
            preference_attribute_feature = self.object_MLP(preference_clip_feature.float()).reshape(bs, -1, self.args.attribute_k2, self.attribute_dim)
            preference_attribute_feature = preference_attribute_feature / preference_attribute_feature.norm(dim=-1, keepdim=True)
            preference_attribute_feature_q_st, preference_attribute_feature_q, idx_preference = self.codebook.straight_through(preference_attribute_feature)

        return ins_attribute_feature, basic_object_attribute_feature, preference_attribute_feature, idx_ins, idx_basic, idx_preference

    def forward_rl_agent(self, ins_encode_feature, rgb_image):
        # ins_encode_feature = self.forward_instruction(ins)
        ins_encode_feature = ins_encode_feature / ins_encode_feature.norm(dim=-1, keepdim=True)
        ins_attribute_feature = self.instruction_MLP(ins_encode_feature).reshape(-1, self.args.attribute_k1, self.attribute_dim)
        ins_attribute_feature = ins_attribute_feature / ins_attribute_feature.norm(dim=-1, keepdim=True)
        if self.args.obj_detector == 'yolo':
            pass
        else:
            obj_clip_feature = self.forward_object(rgb_image).reshape(-1, self.attribute_dim)
            obj_attribute_feature = self.object_MLP(obj_clip_feature).reshape(-1, self.args.attribute_k2, self.attribute_dim)
            obj_attribute_feature = obj_attribute_feature / obj_attribute_feature.norm(dim=-1, keepdim=True)

        return ins_attribute_feature, obj_attribute_feature, obj_clip_feature

    def forward_map(self, ins, obj_list):
        bs = 1
        if isinstance(ins, str):
            ins_encode_feature = self.forward_instruction(ins)
            ins_encode_feature = ins_encode_feature / ins_encode_feature.norm(dim=-1, keepdim=True)
            ins_attribute_feature = self.instruction_MLP(ins_encode_feature).reshape(bs, self.args.attribute_k1, self.attribute_dim)
            ins_attribute_feature = ins_attribute_feature / ins_attribute_feature.norm(dim=-1, keepdim=True)
            obj_clip_feature = self.forward_object(obj_list).reshape(-1, self.attribute_dim)
            obj_attribute_feature = self.object_MLP(obj_clip_feature.float()).reshape(-1, self.args.attribute_k2, self.attribute_dim)
            obj_attribute_feature = obj_attribute_feature / obj_attribute_feature.norm(dim=-1, keepdim=True)

            ins_attribute_feature_q_st, ins_attribute_feature_q, idx_ins = self.codebook.straight_through(ins_attribute_feature)
            obj_attribute_feature_q_st, obj_attribute_feature_q, idx_obj = self.codebook.straight_through(obj_attribute_feature)
            return ins_attribute_feature, obj_attribute_feature, obj_clip_feature, idx_ins, idx_obj

        if isinstance(ins, tuple):
            basic_ins_attribute_feature = self.forward_attribute(ins[1]).reshape(bs, -1, self.attribute_dim)
            preferred_ins_attribute_feature = self.forward_attribute(ins[2]).reshape(bs, -1, self.attribute_dim)

            obj_clip_feature = self.forward_object(obj_list).reshape(-1, self.attribute_dim)
            obj_attribute_feature = self.object_MLP(obj_clip_feature.float()).reshape(-1, self.args.attribute_k2, self.attribute_dim)
            obj_attribute_feature = obj_attribute_feature / obj_attribute_feature.norm(dim=-1, keepdim=True)

            basic_ins_attribute_feature_q_st, basic_ins_attribute_feature_q, basic_idx_ins = self.codebook.straight_through(basic_ins_attribute_feature)
            preferred_ins_attribute_feature_q_st, preferred_ins_attribute_feature_q, preferred_idx_ins = self.codebook.straight_through(preferred_ins_attribute_feature)
            obj_attribute_feature_q_st, obj_attribute_feature_q, idx_obj = self.codebook.straight_through(obj_attribute_feature)

            return basic_ins_attribute_feature, preferred_ins_attribute_feature, obj_attribute_feature, obj_clip_feature, basic_idx_ins, preferred_idx_ins, idx_obj

    def save_model(self, path):
        exclude_layers = ['ins_encoder_model', "CLIP_model"]
        state_dict = {k: v for k, v in self.state_dict().items() if not any(exclude_layer in k for exclude_layer in exclude_layers)}
        torch.save(state_dict, path)

    def save_model_best(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        # 加载保存的模型状态
        state_dict_saved = torch.load(path)

        # 获取当前模型的状态字典
        state_dict_model = self.state_dict()

        # 更新当前模型状态字典中存在的键
        state_dict_model.update({k: v for k, v in state_dict_saved.items() if k in state_dict_model})

        # for k in state_dict_model.keys():
        #     if k not in state_dict_saved:
        #         print(f"key {k} not found in saved model")

        # 加载更新后的状态字典到模型中
        self.load_state_dict(state_dict_model)
