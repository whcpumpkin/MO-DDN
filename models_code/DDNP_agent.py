import torch
import torch.nn as nn
import clip
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from attributebook.attribute_model import AttributeModel
import time
import json
from habitat.tasks.nav.shortest_path_follower import DDNPlusShortestPathFollower
import random
from copy import deepcopy
import numpy as np
import math
from utils.util import MaskCategorical

device = "cuda" if torch.cuda.is_available() else "cpu"


class ClipImageEncoder():

    def __init__(self, model_name="ViT-L/14", device=None):
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load(model_name, device=self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward(self, image):
        with torch.no_grad():
            image = self.preprocess(image).to(self.device)
            image_features = self.model.encode_image(image)
        return image_features


class SimpleCNN(nn.Module):

    def __init__(self, input_channel=1):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=11, stride=1, padding=5),  # 输入通道1，输出通道16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ).to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return x


class DepthEncoder:

    def __init__(self, model_name='clip', image_size=1024):
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        if model_name == 'clip':
            self.model, self.preprocess = clip.load('ViT-L/14', device=self.device)
            self.preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        elif model_name == 'cnn':
            self.model = SimpleCNN()
            # self.preprocess = transforms.Compose([
            #     transforms.ToTensor(),
            #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])
            x_d = torch.randn(1, 1, image_size, image_size).to(self.device)
            f_d = self.model(x_d)
            mlp_input_size = f_d.size(0) * f_d.size(1)
            self.fc1 = nn.Linear(mlp_input_size, 4096).to(device)  # 需要根据实际情况调整
            self.fc2 = nn.Linear(4096, 768).to(device)

        else:
            raise ValueError("Unsupported model type. Choose 'clip' or 'cnn'.")

    def forward(self, image):

        if self.model_name == 'clip':
            # with torch.no_grad():
            image = image.repeat_interleave(3, dim=1)  # 将单通道图像转换为三通道伪RGB图像

            image_tensor = self.preprocess(image / 255).to(self.device)  # Add batch dimension
            image_features = self.model.encode_image(image_tensor).float()
            image_features = self.additional_linear(image_features)  #升维度用的线性层，没有训练，这里只是凑数

        elif self.model_name == 'cnn':
            # with torch.no_grad():
            # image = image.convert('RGB')
            image_tensor = image.to(device)  # Add batch dimension
            image_features = self.model(image_tensor)
            image_features = F.relu(self.fc1(image_features))
            image_features = self.fc2(image_features)

        return image_features


class InstructionEncoder(nn.Module):

    def __init__(self):
        super(InstructionEncoder, self).__init__()
        self.roberta = AutoModel.from_pretrained('roberta-large')
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    def forward(self, input_instruction, attention_mask=None):
        input_ids = self.tokenizer(input_instruction, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # We use the pooled output by default
        return pooled_output


class AttributeProjection(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttributeProjection, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.fc(x)


class AgentModel(nn.Module):

    def __init__(self, image_size=512, attribute_feature_dim=768, num_actions=7, depth_model_name='cnn', device=None, args=None):
        super(AgentModel, self).__init__()
        self.args = args
        self.device = args.device

        #image and text encoder
        # self.clip_image_encoder = ClipImageEncoder()
        self.attribute_model = AttributeModel(args)
        clip_dim = 1024

        #depth encoder
        self.depth_encoder = DepthEncoder(model_name=depth_model_name, image_size=image_size)

        #cls token
        self.cls_token = nn.Parameter(torch.randn(1, attribute_feature_dim))

        #transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=attribute_feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

        # prev action encoder
        self.prev_action_embedding_layer = nn.Linear(1, args.action_embedding_dim).to(self.device)
        if args.use_gps_compass:
            self.gps_compass_encoder = nn.Linear(3, args.gps_compass_embedding_dim).to(self.device)

        #lstm
        self.lstm = nn.LSTM(input_size=attribute_feature_dim + args.action_embedding_dim + args.gps_compass_embedding_dim, hidden_size=args.rnn_hidden_state_dim, batch_first=True, num_layers=args.lstm_layer_num)
        self.actor_feature = nn.Sequential(nn.Linear(args.rnn_hidden_state_dim, args.rnn_hidden_state_dim), nn.ReLU())
        self.actor = MaskCategorical(args.rnn_hidden_state_dim, self.args.action_space)
        self.critic = nn.Sequential(nn.Linear(args.rnn_hidden_state_dim, args.rnn_hidden_state_dim), nn.ReLU(), nn.Linear(args.rnn_hidden_state_dim, 1))
        self.use_clipped_value_loss = True

    def forward(self, rgb_image, depth_image, ddnplusgoal_feature_input, other_inputs=None, deterministic=False):
        # rgb_image = torch.tensor(rgb_image / 255, dtype=torch.float32)
        # depth_image = torch.tensor(depth_image, dtype=torch.float32)
        if len(rgb_image.shape) == 3:
            rgb_image = rgb_image.unsqueeze(0)
        if len(depth_image.shape) == 3:
            depth_image = depth_image.unsqueeze(0)
        rgb_image = rgb_image.permute(0, 3, 1, 2) / 255.0
        rgb_image = rgb_image.to(torch.float16)
        depth_image = depth_image.permute(0, 3, 1, 2).to(torch.float32)

        batch_size = rgb_image.size(0)
        ins_attribute_feature, obj_attribute_feature, obj_clip_feature = self.attribute_model.forward_rl_agent(ddnplusgoal_feature_input, rgb_image)
        obj_clip_feature = obj_clip_feature.unsqueeze(1)
        ddnplusgoal_feature_input = ddnplusgoal_feature_input.unsqueeze(1)

        depth_feature = self.depth_encoder.forward(depth_image.to(self.device)).unsqueeze(1)

        cls_token_expand = self.cls_token.expand(batch_size, -1).unsqueeze(1)

        transformer_input = torch.cat((cls_token_expand, obj_clip_feature, obj_attribute_feature, ddnplusgoal_feature_input, ins_attribute_feature, depth_feature), dim=1)

        transformer_output = self.transformer_encoder(transformer_input)
        cls_output = transformer_output[:, 0, :]
        prev_action_embedding = self.prev_action_embedding_layer(other_inputs["prev_action"])
        if self.args.use_gps_compass:
            gps_compass_embedding = self.gps_compass_encoder(other_inputs["gps_compass"].to(self.device))
            lstm_input = torch.cat((cls_output, prev_action_embedding, gps_compass_embedding), dim=1).unsqueeze(1)
        else:
            lstm_input = torch.cat((cls_output, prev_action_embedding), dim=1).unsqueeze(1)
        lstm_output, hidden_state = self.lstm(lstm_input, (other_inputs['prev_hidden_h'], other_inputs['prev_hidden_c']))
        #lstm_output, _ = self.lstm(transformer_output.last_hidden_state)
        actor_feature_input = self.actor_feature(lstm_output)
        # action_mask = torch.zeros(batch_size, 1, self.args.action_space).to(self.device)
        # action_mask[:, :, 0] = -float('inf')
        action_dist = self.actor(actor_feature_input)
        if deterministic:
            action_out = action_dist.mode()
        else:
            action_out = action_dist.sample()
        value = self.critic(lstm_output)

        return action_out, action_dist, value, hidden_state, transformer_input

    def act(self, rgb_input, depth_input, ddnplusgoal_feature_input, other_inputs=None, deterministic=False):
        action_out, action_dist, value, hidden_state, check_input = self.forward_il_single(rgb_input, depth_input, ddnplusgoal_feature_input, other_inputs, deterministic=deterministic)
        action_log_probs = action_dist.log_probs(action_out)
        action_dist_entropy = action_dist.entropy().mean()
        return action_out, action_dist, value, action_log_probs, action_dist_entropy, hidden_state

    def act_batch(self, rgb_input, depth_input, ddnplusgoal_feature_input, masks, action, other_inputs=None):
        num_envs_per_batch = self.args.workers // self.args.num_mini_batch
        seq_len = int(other_inputs['prev_action'].shape[0] // num_envs_per_batch)
        value, action_out, action_dist, hx, cx = self.forward_batch(rgb_input, depth_input, ddnplusgoal_feature_input, other_inputs, masks)

        action_log_probs = action_dist.log_probs(action.to(self.args.device))
        action_dist_entropy = action_dist.entropy().mean()
        return value, action_log_probs, action_dist_entropy, hx, cx

    def forward_batch(self, rgb_image, depth_image, ddnplusgoal_feature_input, other_inputs, masks, pre_train_Q=False):
        num_envs_per_batch = self.args.workers // self.args.num_mini_batch
        seq_len = int(other_inputs['prev_action'].shape[0] // num_envs_per_batch)

        rgb_image = rgb_image.permute(0, 3, 1, 2) / 255.0
        rgb_image = rgb_image.to(torch.float16)
        depth_image = depth_image.permute(0, 3, 1, 2).to(torch.float32)

        ins_attribute_feature, obj_attribute_feature, obj_clip_feature = self.attribute_model.forward_rl_agent(ddnplusgoal_feature_input, rgb_image)
        text_features = ddnplusgoal_feature_input.unsqueeze(1)
        obj_clip_feature = obj_clip_feature.unsqueeze(1)

        depth_feature = self.depth_encoder.forward(depth_image).unsqueeze(1)

        cls_token_expand = self.cls_token.expand(seq_len * num_envs_per_batch, -1).unsqueeze(1)

        transformer_input = torch.cat((cls_token_expand, obj_clip_feature, obj_attribute_feature, text_features, ins_attribute_feature, depth_feature), dim=1)

        transformer_output = self.transformer_encoder(transformer_input)
        cls_output = transformer_output[:, 0, :]
        prev_action_embedding = self.prev_action_embedding_layer(other_inputs["prev_action"].float())
        if self.args.use_gps_compass:
            gps_compass_embedding = self.gps_compass_encoder(other_inputs["gps_compass"].to(self.device))
            t_feature = torch.cat((cls_output, prev_action_embedding, gps_compass_embedding), dim=-1).reshape(num_envs_per_batch, seq_len, -1)
        else:
            t_feature = torch.cat((cls_output, prev_action_embedding), dim=-1).reshape(num_envs_per_batch, seq_len, -1)

        seq_len = t_feature.shape[1]
        masks = masks.reshape(num_envs_per_batch, seq_len)

        has_zeros = ((masks[:, 1:] == 0.0).any(dim=0).nonzero().squeeze().cpu())
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()
        if isinstance(has_zeros, list) is False:
            print(has_zeros)
            print(masks.shape)
            print(masks)
        has_zeros = [0] + has_zeros + [seq_len]
        outputs = []

        hx = other_inputs["prev_hidden_h"].permute(1, 0, 2).to(self.args.device).contiguous()
        cx = other_inputs["prev_hidden_c"].permute(1, 0, 2).to(self.args.device).contiguous()
        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]
            output, (hx, cx) = self.lstm(t_feature.contiguous()[:, start_idx:end_idx, :], (hx * masks[:, start_idx].reshape(1, -1, 1), cx * masks[:, start_idx].reshape(1, -1, 1)))
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        # x = output.reshape([1, self.args.rnn_hidden_state_dim])
        # action_dist = self.action(output).reshape(seq_len*self.args.workers, -1)
        actor_feature_input = self.actor_feature(outputs.reshape(seq_len * num_envs_per_batch, -1))
        # action_mask = torch.zeros(num_envs_per_batch * seq_len, self.args.action_space).to(self.device)
        # action_mask[:, 0] = -float('inf')
        action_dist = self.actor(actor_feature_input)
        action_out = action_dist.sample()
        value = self.critic(outputs).reshape(seq_len * num_envs_per_batch, -1)
        return value, action_out, action_dist, hx, cx

    def forward_il(self, rgb_image, depth_image, ddnplusgoal_feature_input, prev_action, gps_compass=None, other_inputs=None):

        # time_start = time.time()
        traj_seq_len = rgb_image.shape[0]
        w, h = rgb_image.shape[1], rgb_image.shape[2]

        prev_action = torch.cat((torch.tensor([1]), prev_action), dim=0)
        prev_action = prev_action[:-1]
        prev_action = prev_action.float().reshape(traj_seq_len, -1).to(self.device)
        # print("pre time:", time.time() - time_start)

        rgb_image = rgb_image.permute(0, 3, 1, 2) / 255.0
        rgb_image = rgb_image.to(torch.float16)
        depth_image = depth_image.permute(0, 3, 1, 2).to(torch.float32)

        # time_start = time.time()
        ddnplusgoal_feature_input = ddnplusgoal_feature_input.unsqueeze(1).repeat(1, traj_seq_len, 1).reshape(traj_seq_len, -1)
        ins_attribute_feature, obj_attribute_feature, obj_clip_feature = self.attribute_model.forward_rl_agent(ddnplusgoal_feature_input, rgb_image)
        # print("attribute time:", time.time() - time_start)

        obj_clip_feature = obj_clip_feature.unsqueeze(1)
        ddnplusgoal_feature_input = ddnplusgoal_feature_input.unsqueeze(1)

        # time_start = time.time()
        depth_feature = self.depth_encoder.forward(depth_image.to(self.device)).unsqueeze(1)
        # print("depth time:", time.time() - time_start)

        # time_start = time.time()
        cls_token_expand = self.cls_token.expand(traj_seq_len, -1).unsqueeze(1)

        transformer_input = torch.cat((cls_token_expand, obj_clip_feature, obj_attribute_feature, ddnplusgoal_feature_input, ins_attribute_feature, depth_feature), dim=1)
        transformer_output = self.transformer_encoder(transformer_input)

        cls_output = transformer_output[:, 0, :]
        # print("transformer time:", time.time() - time_start)

        # time_start = time.time()
        prev_action_embedding = self.prev_action_embedding_layer(prev_action)
        # print("prev_action_embedding time:", time.time() - time_start)

        if self.args.use_gps_compass:
            # time_start = time.time()
            gps_compass_embedding = self.gps_compass_encoder(gps_compass).reshape(traj_seq_len, -1)
            lstm_input = torch.cat((cls_output, prev_action_embedding, gps_compass_embedding), dim=1).unsqueeze(0).reshape(traj_seq_len, -1)
            # print("gps_compass_embedding forward time:", time.time() - time_start)
        else:
            lstm_input = torch.cat((cls_output, prev_action_embedding), dim=1).unsqueeze(0).reshape(traj_seq_len, -1)
        lstm_input = lstm_input.reshape(1, traj_seq_len, -1)
        # time_start = time.time()
        lstm_output, hidden_state = self.lstm(lstm_input)
        # print("lstm forward time:", time.time() - time_start)
        # time_start = time.time()
        actor_feature_input = self.actor_feature(lstm_output)
        # print("actor_feature_input forward time:", time.time() - time_start)

        # time_start = time.time()
        # action_mask = torch.zeros(batch_size, traj_seq_len, self.args.action_space).to(self.device)
        # action_mask[:, :, 0] = -999999999
        # print("action_mask time:", time.time() - time_start)

        # time_start = time.time()
        action_dist = self.actor(actor_feature_input)
        # print("actor forward time:", time.time() - time_start)
        return action_dist

    def forward_il_single(self, rgb_image, depth_image, ddnplusgoal_feature_input, other_inputs=None, deterministic=False):

        # time_start = time.time()
        if type(rgb_image) == np.ndarray:
            rgb_image = torch.from_numpy(rgb_image).float().to(self.device)
        if type(depth_image) == np.ndarray:
            depth_image = torch.from_numpy(depth_image).float().to(self.device)
        if type(ddnplusgoal_feature_input) == list:
            ddnplusgoal_feature_input = torch.from_numpy(np.array(ddnplusgoal_feature_input)).float().to(self.device)

        if len(rgb_image.shape) == 3:
            rgb_image = rgb_image.unsqueeze(0)
        if len(depth_image.shape) == 3:
            depth_image = depth_image.unsqueeze(0)
        bs = rgb_image.shape[0]
        traj_seq_len = 1
        w, h = rgb_image.shape[1], rgb_image.shape[2]

        # prev_action = torch.cat((torch.tensor([1]), prev_action), dim=0)
        # prev_action = prev_action[:-1]
        # prev_action = prev_action.float().reshape(traj_seq_len, -1).to(self.device)
        # print("pre time:", time.time() - time_start)

        rgb_image = rgb_image.permute(0, 3, 1, 2) / 255.0
        rgb_image = rgb_image.to(torch.float16)
        depth_image = depth_image.permute(0, 3, 1, 2).to(torch.float32)

        # time_start = time.time()
        ddnplusgoal_feature_input = ddnplusgoal_feature_input.unsqueeze(1).repeat(1, traj_seq_len, 1).reshape(bs, -1)
        ins_attribute_feature, obj_attribute_feature, obj_clip_feature = self.attribute_model.forward_rl_agent(ddnplusgoal_feature_input, rgb_image)
        # print("attribute time:", time.time() - time_start)

        obj_clip_feature = obj_clip_feature.unsqueeze(1)
        ddnplusgoal_feature_input = ddnplusgoal_feature_input.unsqueeze(1)

        # time_start = time.time()
        depth_feature = self.depth_encoder.forward(depth_image.to(self.device)).unsqueeze(1)
        # print("depth time:", time.time() - time_start)

        # time_start = time.time()
        cls_token_expand = self.cls_token.expand(bs, -1).unsqueeze(1)

        transformer_input = torch.cat((cls_token_expand, obj_clip_feature, obj_attribute_feature, ddnplusgoal_feature_input, ins_attribute_feature, depth_feature), dim=1)
        transformer_output = self.transformer_encoder(transformer_input)

        cls_output = transformer_output[:, 0, :]
        # print("transformer time:", time.time() - time_start)

        # time_start = time.time()
        prev_action_embedding = self.prev_action_embedding_layer(other_inputs["prev_action"])
        # print("prev_action_embedding time:", time.time() - time_start)

        if self.args.use_gps_compass:
            # time_start = time.time()
            gps_compass_embedding = self.gps_compass_encoder(other_inputs['gps_compass'].to(self.device)).reshape(bs, -1)
            lstm_input = torch.cat((cls_output, prev_action_embedding, gps_compass_embedding), dim=1).unsqueeze(0).reshape(bs, traj_seq_len, -1)
            # print("gps_compass_embedding forward time:", time.time() - time_start)
        else:
            lstm_input = torch.cat((cls_output, prev_action_embedding), dim=1).unsqueeze(0).reshape(traj_seq_len, -1)
        # time_start = time.time()
        lstm_output, hidden_state = self.lstm(lstm_input, (other_inputs['prev_hidden_h'], other_inputs['prev_hidden_c']))
        # print("lstm forward time:", time.time() - time_start)
        # time_start = time.time()
        actor_feature_input = self.actor_feature(lstm_output)
        # print("actor_feature_input forward time:", time.time() - time_start)

        # time_start = time.time()
        # action_mask = torch.zeros(batch_size, traj_seq_len, self.args.action_space).to(self.device)
        # action_mask[:, :, 0] = -999999999
        # print("action_mask time:", time.time() - time_start)

        # time_start = time.time()
        action_dist = self.actor(actor_feature_input)
        # print("actor forward time:", time.time() - time_start)
        if deterministic:
            action_out = action_dist.mode()
        else:
            action_out = action_dist.sample()
        value = self.critic(lstm_output)

        return action_out, action_dist, value, hidden_state, transformer_input

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    pass
