import numpy as np
import os
import time
import json

import torch
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.my_dataloader import Attribute_Dataset, my_collate_fn, ddnp_collate_fn, DDNP_Dataset
from utils.args import parse_arguments
from attributebook.attribute_model import AttributeModel
from torch.utils.tensorboard import SummaryWriter
import statistics
from scipy.stats import entropy


def load_coodebook():
    #load codebook
    codebook = np.load('attributebook/cluster_centers.npy')
    # import as nn.Embedding
    codebook_embedding = torch.nn.Embedding(codebook.shape[0], codebook.shape[1])
    codebook_embedding.weight.data.copy_(torch.from_numpy(codebook))
    t = 1


def check_attribute():
    path_to_instruction = "attributebook/attribute_merge_checked.json"
    with open(path_to_instruction, 'r') as f:
        data_instruction = json.load(f)
    path_to_object = "attributebook/obj_attribute_checked.json"
    with open(path_to_object, 'r') as f:
        data_object = json.load(f)

    path_to_label_map = "attributebook/label_map.json"
    with open(path_to_label_map, 'r') as f:
        label_map = json.load(f)

    all_match = 0
    right_match = 0
    not_in_data_object = 0
    for instruction_id in tqdm(data_instruction.keys()):
        attribute_list_instruction = data_instruction[instruction_id][0]
        attribute_idx_list_instruction = [label_map[attribute] for attribute in attribute_list_instruction]
        object_list = data_instruction[instruction_id][1]
        for solution in object_list:
            object_attribute_for_solution = []
            for object in solution:
                # if object in data_object.keys():
                object_attribute_for_solution.extend(data_object[object])
                # else:
                #     object_ = object.replace("_", " ")
                #     if object_ in data_object.keys():
                #         object_attribute_for_solution.extend(data_object[object_])
                #         data_object[object] = data_object[object_]
                #         del data_object[object_]

                #     else:
                #         print(object)

                # not_in_data_object += 1
            object_attribute_idx_for_solution = [label_map[attribute] for attribute in object_attribute_for_solution]
            all_match += 1
            for attribute_idx in attribute_idx_list_instruction:
                if attribute_idx in object_attribute_idx_for_solution:
                    right_match += 1.0 / len(attribute_idx_list_instruction)
    #save data_object
    # with open('attributebook/obj_attribute_checked.json', 'w') as f:
    #     json.dump(data_object, f, indent=4)
    print("all_match:", all_match)
    print("right_match:", right_match)
    print("accuracy:", right_match / all_match)
    print("not_in_data_object:", not_in_data_object)


def test_attribute_model(attribute_model, args, writer, epoch=0):
    task_dataset = DDNP_Dataset(args)
    task_loader = DataLoader(task_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=ddnp_collate_fn)
    attribute_model.eval()
    bs = 1
    basic_mse_loss_sum = 0
    preference_mse_loss_sum = 0
    ins_idx_dict = {}
    basic_idx_dict = {}
    preference_idx_dict = {}
    for i in range(args.attributebook_size):
        ins_idx_dict[i] = 0
        basic_idx_dict[i] = 0
        preference_idx_dict[i] = 0
    idx_list = []
    cover_basic = 0
    cover_preference = 0
    cover_all = 0
    for i, batch in tqdm(enumerate(task_loader), total=len(task_loader)):
        ins, basic_object, preference_object, basic_demand_instruction, preference_demand_instruction = batch
        with torch.no_grad():
            ins_attribute_feature, basic_object_attribute_feature, preference_attribute_feature, idx_ins, idx_basic, idx_preference = attribute_model.forward_test((ins, basic_object, preference_object))
        idx_ins_list = idx_ins.tolist()
        if idx_basic is not None:
            idx_basic_list = idx_basic.tolist()
        else:
            idx_basic_list = []
        if idx_preference is not None:
            idx_preference_list = idx_preference.tolist()
        else:
            idx_preference_list = []
        idx_basic_list.extend(idx_preference_list)
        idx_list.append((idx_basic_list, idx_preference_list, idx_ins_list))

        for idx in idx_ins_list:
            if idx not in ins_idx_dict.keys():
                ins_idx_dict[idx] = 0
            ins_idx_dict[idx] += 1
            if idx in idx_basic_list:
                cover_basic += 1
            if idx in idx_preference_list:
                cover_preference += 1
            cover_all += 1
        for idx in idx_basic_list:
            if idx not in basic_idx_dict.keys():
                basic_idx_dict[idx] = 0
            basic_idx_dict[idx] += 1
        for idx in idx_preference_list:
            if idx not in preference_idx_dict.keys():
                preference_idx_dict[idx] = 0
            preference_idx_dict[idx] += 1

        # Instruction-BasicObject matching loss

        if basic_object_attribute_feature is not None:
            k3 = basic_object_attribute_feature.shape[1]
            ins_attribute_feature_match = ins_attribute_feature.unsqueeze(1)  # [batch_size, 1, attribute_num , attribute_dim]
            ins_attribute_feature_match = ins_attribute_feature_match.repeat(1, k3, 1, 1).reshape(bs * k3, args.attribute_k1, args.transformer_embedding_dim)  # [batch_size*obj_num, attribute_num, attribute_dim]
            basic_object_attribute_feature = basic_object_attribute_feature.reshape(bs * k3, args.attribute_k2, args.transformer_embedding_dim)  # [batch_size*obj_num, attribute_num, attribute_dim]
            ins_attribute_feature_match_expanded = ins_attribute_feature_match.unsqueeze(2).expand(-1, -1, basic_object_attribute_feature.size(1), -1)
            basic_object_attribute_feature_expanded = basic_object_attribute_feature.unsqueeze(1).expand(-1, ins_attribute_feature_match.size(1), -1, -1)
            basic_mse_loss = ((ins_attribute_feature_match_expanded - basic_object_attribute_feature_expanded)**2).mean(dim=-1)  # [batch_size*obj_num, attribute_num]
            basic_mse_loss = basic_mse_loss.min(dim=2)[0].min(dim=1)[0].mean()
        else:
            basic_mse_loss = torch.tensor(0.0).to(args.device)

        # Instruction-Preference matching loss

        if preference_attribute_feature is not None:
            k3 = preference_attribute_feature.shape[1]
            ins_attribute_feature_match = ins_attribute_feature.unsqueeze(1)  # [batch_size, 1, attribute_num , attribute_dim]
            ins_attribute_feature_match = ins_attribute_feature_match.repeat(1, k3, 1, 1).reshape(bs * k3, args.attribute_k1, args.transformer_embedding_dim)  # [batch_size*obj_num, attribute_num, attribute_dim]
            preference_attribute_feature = preference_attribute_feature.reshape(bs * k3, args.attribute_k2, args.transformer_embedding_dim)  # [batch_size*obj_num, attribute_num, attribute_dim]
            ins_attribute_feature_match_expanded = ins_attribute_feature_match.unsqueeze(2).expand(-1, -1, preference_attribute_feature.size(1), -1)
            preference_attribute_feature_expanded = preference_attribute_feature.unsqueeze(1).expand(-1, ins_attribute_feature_match.size(1), -1, -1)
            preference_mse_loss = ((ins_attribute_feature_match_expanded - preference_attribute_feature_expanded)**2).mean(dim=-1)  # [batch_size*obj_num, attribute_num]
            preference_mse_loss = preference_mse_loss.min(dim=2)[0].min(dim=1)[0].mean()
        else:
            preference_mse_loss = torch.tensor(0.0).to(args.device)
        basic_mse_loss_sum += basic_mse_loss.item()
        preference_mse_loss_sum += preference_mse_loss.item()
    other_cover_basic = 0
    other_cover_preference = 0
    other_cover_all = 0
    for i in range(len(idx_list)):
        idx_basic_list_i, idx_preference_list_i, idx_ins_list_i = idx_list[i]
        for j in range(len(idx_list)):
            if i == j:
                continue
            idx_basic_list_j, idx_preference_list_j, idx_ins_list_j = idx_list[j]
            for idx in idx_ins_list_i:
                if idx in idx_basic_list_j:
                    other_cover_basic += 1
                if idx in idx_preference_list_j:
                    other_cover_preference += 1
                other_cover_all += 1

    ins_entropy = entropy(list(ins_idx_dict.values()))
    basic_entropy = entropy(list(basic_idx_dict.values()))
    preference_entropy = entropy(list(preference_idx_dict.values()))

    print("cover_basic_rate: {}, cover_preference_rate: {}".format(cover_basic / cover_all, cover_preference / cover_all))
    print("other_cover_basic_rate: {}, other_cover_preference_rate: {}".format(other_cover_basic / other_cover_all, other_cover_preference / other_cover_all))
    print("basic_mse_loss_sum: {}, preference_mse_loss_sum: {}".format(basic_mse_loss_sum, preference_mse_loss_sum))

    print("ins_entropy: {}, basic_entropy: {}, preference_entropy: {}".format(ins_entropy, basic_entropy, preference_entropy))

    writer.add_scalar('test/mse_loss_sum_basic_', basic_mse_loss_sum, epoch)
    writer.add_scalar('test/mse_loss_sum_preference', preference_mse_loss_sum, epoch)

    writer.add_scalar('test/rate_cover_basic_', cover_basic / cover_all, epoch)
    writer.add_scalar('test/rate_cover_preference', cover_preference / cover_all, epoch)

    writer.add_scalar('test/rate_other_cover_basic', other_cover_basic / other_cover_all, epoch)
    writer.add_scalar('test/rate_other_cover_preference', other_cover_preference / other_cover_all, epoch)
    writer.add_scalar('test/rate_delta_cover_basic', cover_basic / cover_all - other_cover_basic / other_cover_all, epoch)
    writer.add_scalar('test/rate_delta_cover_preference', cover_preference / cover_all - other_cover_preference / other_cover_all, epoch)

    writer.add_scalar('test/entropy_ins', ins_entropy, epoch)
    writer.add_scalar('test/entropy_basic', basic_entropy, epoch)
    writer.add_scalar('test/entropy_preference', preference_entropy, epoch)

    return (cover_basic / cover_all) - (other_cover_basic / other_cover_all)


def train(args):
    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_path = os.path.join(args.save_dir, args.save_name, time_now)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    #save args
    with open(os.path.join(save_path, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    writer = SummaryWriter(log_dir=save_path)
    attribute_dataset = Attribute_Dataset(args)
    attribute_loader = DataLoader(attribute_dataset, batch_size=args.attribute_batch_size, shuffle=True, num_workers=16, collate_fn=my_collate_fn)
    attribute_model = AttributeModel(args)
    attribute_model.to(args.device)
    optimizer = torch.optim.Adam(attribute_model.parameters(), lr=args.attribute_lr)
    max_save_rate = -9999
    attribute_model.save_model(os.path.join(save_path, "attribute_model_{}.pth".format(0)))
    save_rate = test_attribute_model(attribute_model, args, writer, 0)
    if save_rate > max_save_rate:
        attribute_model.save_model_best(os.path.join(save_path, "attribute_model_best.pth"))
        max_save_rate = save_rate
    for epoch in range(args.attribute_epoch):
        for i, batch in tqdm(enumerate(attribute_loader), total=len(attribute_loader)):
            bs = len(batch[0])

            ins_bge_feature, ins_attribute_feature, obj_clip_feature, obj_attribute_feature, ins_attribute_feature_gt, obj_attribute_feature_gt, recon_ins_bge_feature, recon_obj_clip_feature, ins_attribute_feature_q_st, obj_attribute_feature_q_st, ins_attribute_feature_q, obj_attribute_feature_q = attribute_model(
                batch)

            # Reconstruction loss
            recon_loss = F.mse_loss(ins_bge_feature, recon_ins_bge_feature) + F.mse_loss(obj_clip_feature, recon_obj_clip_feature)

            # Vector quantization objective
            vq_loss = F.mse_loss(ins_attribute_feature_q.detach(), ins_attribute_feature) + F.mse_loss(obj_attribute_feature_q.detach(), obj_attribute_feature)

            # Commitment objective
            commit_loss = F.mse_loss(ins_attribute_feature_q, ins_attribute_feature.detach()) + F.mse_loss(obj_attribute_feature_q, obj_attribute_feature.detach())
            loss = args.recon_coef * recon_loss + args.vq_coef * vq_loss + args.commit_coef * commit_loss
            if args.loss_type == "mse":

                # Attribute loss
                ins_attribute_loss = F.mse_loss(ins_attribute_feature_gt, ins_attribute_feature)
                obj_attribute_loss = F.mse_loss(obj_attribute_feature_gt, obj_attribute_feature)
                attribute_loss = ins_attribute_loss + obj_attribute_loss

                # Push Loss
                random_indices_bs = torch.randperm(bs)
                shuffled_ins_attribute_feature = ins_attribute_feature[random_indices_bs]
                random_perms_k1 = torch.stack([torch.randperm(args.attribute_k1, device=shuffled_ins_attribute_feature.device) for _ in range(bs)])
                shuffled_ins_attribute_feature = shuffled_ins_attribute_feature.gather(1, random_perms_k1.unsqueeze(2).repeat(1, 1, args.transformer_embedding_dim))

                # 打乱batch_size维度
                batch_perm = torch.randperm(bs, device=shuffled_ins_attribute_feature.device)
                obj_attribute_feature_batch_shuffled = obj_attribute_feature[batch_perm]

                # 为k3和k1维度生成随机排列，对每个样本独立进行
                k3_perms = torch.stack([torch.randperm(args.attribute_k3, device=shuffled_ins_attribute_feature.device) for _ in range(bs)])
                k1_perms = torch.stack([torch.randperm(args.attribute_k1, device=shuffled_ins_attribute_feature.device) for _ in range(bs)])

                # 扩展k3和k1的随机排列以匹配tensor的维度
                k3_perms_expanded = k3_perms.unsqueeze(2).unsqueeze(3).repeat(1, 1, args.attribute_k1, args.transformer_embedding_dim)
                k1_perms_expanded = k1_perms.unsqueeze(1).unsqueeze(3).repeat(1, args.attribute_k3, 1, args.transformer_embedding_dim)

                # 使用gather函数应用打乱
                obj_attribute_feature_k3_shuffled = obj_attribute_feature_batch_shuffled.gather(1, k3_perms_expanded)
                obj_attribute_feature_k1_k3_shuffled = obj_attribute_feature_k3_shuffled.gather(2, k1_perms_expanded)

                push_ins_loss = -F.mse_loss(ins_attribute_feature, shuffled_ins_attribute_feature)
                push_obj_loss = -F.mse_loss(obj_attribute_feature, obj_attribute_feature_k1_k3_shuffled)
                loss+=args.obj_attribute_coef * obj_attribute_loss + args.ins_attribute_coef * ins_attribute_loss\
                + args.push_ins_coef * push_ins_loss    + args.push_obj_coef * push_obj_loss
                writer.add_scalar('loss_ins_attribute', ins_attribute_loss.item(), epoch * len(attribute_loader) + i)
                writer.add_scalar('loss_obj_attribute', obj_attribute_loss.item(), epoch * len(attribute_loader) + i)

                writer.add_scalar('loss_push_ins', push_ins_loss.item(), epoch * len(attribute_loader) + i)
                writer.add_scalar('loss_push_obj', push_obj_loss.item(), epoch * len(attribute_loader) + i)

            # Instruction-Object matching loss
            ins_attribute_feature_match = ins_attribute_feature.unsqueeze(1)  # [batch_size, 1, attribute_num , attribute_dim]
            ins_attribute_feature_match = ins_attribute_feature_match.repeat(1, args.attribute_k3, 1, 1).reshape(bs * args.attribute_k3, args.attribute_k1, args.transformer_embedding_dim)  # [batch_size*obj_num, attribute_num, attribute_dim]
            obj_attribute_feature = obj_attribute_feature.reshape(bs * args.attribute_k3, args.attribute_k2, args.transformer_embedding_dim)  # [batch_size*obj_num, attribute_num, attribute_dim]
            ins_attribute_feature_match_expanded = ins_attribute_feature_match.unsqueeze(2).expand(-1, -1, obj_attribute_feature.size(1), -1)
            obj_attribute_feature_expanded = obj_attribute_feature.unsqueeze(1).expand(-1, ins_attribute_feature_match.size(1), -1, -1)
            mse_loss = ((ins_attribute_feature_match_expanded - obj_attribute_feature_expanded)**2).mean(dim=-1)  # [batch_size*obj_num, attribute_num]
            min_loss = mse_loss.min(dim=2)[0].min(dim=1)[0]
            matching_loss = min_loss.mean()

            loss += args.matching_coef * matching_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: {}, Loss: {:.6f}, Recon Loss: {:.6f}, VQ Loss: {:.6f}, Commit Loss: {:.6f}, Matching Loss: {:.6f}".format(epoch, loss.item(), recon_loss.item(), vq_loss.item(), commit_loss.item(), matching_loss.item()))
            writer.add_scalar('loss', loss.item(), epoch * len(attribute_loader) + i)
            writer.add_scalar('loss_recon', recon_loss.item(), epoch * len(attribute_loader) + i)
            writer.add_scalar('loss_vq', vq_loss.item(), epoch * len(attribute_loader) + i)
            writer.add_scalar('loss_commit', commit_loss.item(), epoch * len(attribute_loader) + i)

            writer.add_scalar('loss_matching', matching_loss.item(), epoch * len(attribute_loader) + i)
        save_rate = test_attribute_model(attribute_model, args, writer, epoch + 1)
        attribute_model.save_model(os.path.join(save_path, "attribute_model_{}.pth".format(epoch + 1)))
        if save_rate > max_save_rate:
            attribute_model.save_model_best(os.path.join(save_path, "attribute_model_best.pth"))
            max_save_rate = save_rate


if __name__ == '__main__':
    args = parse_arguments()
    train(args)
