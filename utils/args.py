import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    # Save

    parser.add_argument('--save_dir', type=str, default='LLM+Fine', help='directory to save models')
    parser.add_argument('--save_interval', type=int, default=50, help='save interval')
    parser.add_argument('--save_name', type=str, default='LLM+Fine_stss', help='save name')

    # Eval
    parser.add_argument('--task_mode', type=str, default='train', help='train val')
    parser.add_argument('--scene_mode', type=str, default='train', help='train val')
    parser.add_argument('--random_fine', type=int, default=0, help='random fine')
    parser.add_argument('--add_noise', type=int, default=0, help='add noise')

    parser.add_argument('--FBE', type=int, default=0, help='train val')
    parser.add_argument('--debug', type=int, default=0, help='train val')
    parser.add_argument('--ins_attribute_encoder', type=str, default="attribute", help='train val')

    # Env
    parser.add_argument('--running_mode', type=str, default="train", help='train, val, test')
    parser.add_argument('--dataset_mode', type=str, default="train", help='train, val, test')
    parser.add_argument('--max_step', type=int, default=300, help="max steps in an episode")
    parser.add_argument('--epoch', type=int, default=100, help="max steps in an episode")
    parser.add_argument('--workers', type=int, default=1, help="workers for env")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--device', type=str, default="cuda", help="device")
    parser.add_argument('--task_file', type=str, default='dataset/unfold_task_checked_scene.json', help="task file")
    parser.add_argument('--scene_change_interval', type=int, default=5, help="scene change interval")

    # Agent
    parser.add_argument('--input_image_size', type=int, default=512, help="input image size")
    parser.add_argument('--action_embedding_dim', type=int, default=64, help="action embedding size")
    parser.add_argument("--rnn_hidden_state_dim", type=int, default=1024, help="rnn hidden state dim")
    parser.add_argument("--config_name", type=str, default="config.yaml", help="config file name")
    parser.add_argument("--action_space", type=int, default=7, help="action space without stop")
    parser.add_argument("--lstm_layer_num", type=int, default=2, help="lstm layer num")
    parser.add_argument("--transformer_embedding_dim", type=int, default=768, help="transformer embedding dim")
    parser.add_argument("--transformer_nhead", type=int, default=8, help="transformer nhead")
    parser.add_argument("--transformer_num_layers", type=int, default=6, help="transformer num layers")
    parser.add_argument("--instruction_dim", type=int, default=768, help="instruction_dim")
    parser.add_argument("--ins_model_name", type=str, default='clip', help="instruction encoder model name")
    parser.add_argument('--obj_detector', type=str, default=None, help="obj detector model name")
    parser.add_argument('--eval_model_path', type=str, default="DDNP_save/il_agent/2024-04-03-12-56-44/2024-04-03-12-56-44", help="eval model path")
    parser.add_argument('--use_gps_compass', type=bool, default=True, help="use gps compass")
    parser.add_argument('--gps_compass_embedding_dim', type=int, default=32, help="gps compass embedding dim")
    parser.add_argument('--grad_acc_steps', type=int, default=4, help="grad accumulation steps")
    parser.add_argument("--visual_encoder_type", type=str, default='transformer', help="visual encoder type")

    # RL
    parser.add_argument("--use_linear_lr_decay", type=bool, default=True, help="use linear lr decay")
    parser.add_argument("--use_gae", type=bool, default=True, help="use gae")
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_proper_time_limits', action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument('--ppo_epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=int, default=4, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--rl_lr', type=float, default=2.5e-4, help='learning rate for rl (default: 2.5e-4)')
    parser.add_argument('--rl_finetune', type=bool, default=True, help='finetune rl')
    parser.add_argument('--value_pretrain_epoch', type=int, default=5000, help='value pretrain epochs')
    parser.add_argument('--warmup_epoch', type=int, default=10000, help='warmup epochs')
    parser.add_argument('--il_pretrained_path', type=str, default='attribute_model_4.pth', help='il pretrained path')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')

    # Attribute
    parser.add_argument('--attribute_k1', type=int, default=4, help="attribute embedding size")
    parser.add_argument('--attribute_k2', type=int, default=4, help="attribute embedding size")
    parser.add_argument('--attribute_k3', type=int, default=10, help="attribute embedding size")
    parser.add_argument('--attributebook_size', type=int, default=128, help="attributebook_size")
    parser.add_argument('--path_to_attribute_ins_file', type=str, default='attributebook/attribute_merge_checked.json', help="path to attribute ins file")
    parser.add_argument('--path_to_attribute_object_file', type=str, default='attributebook/obj_attribute_checked.json', help="path to attribute word file")
    parser.add_argument('--path_to_init_codebook_file', type=str, default='attributebook/cluster_centers.npy', help="path to attribute word file")
    parser.add_argument('--attribute_lr', type=float, default=1e-5, help="attribute learning rate")
    parser.add_argument('--attribute_batch_size', type=int, default=64, help="attribute batch size")
    parser.add_argument('--attribute_epoch', type=int, default=5, help="max steps in an episode")
    parser.add_argument('--recon_coef', type=float, default=1.0, help="attribute loss weight")
    parser.add_argument('--vq_coef', type=float, default=1.0, help="attribute loss weight")
    parser.add_argument('--commit_coef', type=float, default=1.0, help="attribute loss weight")
    parser.add_argument('--obj_attribute_coef', type=float, default=1.0, help="attribute loss weight")
    parser.add_argument('--ins_attribute_coef', type=float, default=1.0, help="attribute loss weight")
    parser.add_argument('--matching_coef', type=float, default=1.0, help="attribute loss weight")
    parser.add_argument('--push_ins_coef', type=float, default=1.0, help="attribute loss weight")
    parser.add_argument('--push_obj_coef', type=float, default=1.0, help="attribute loss weight")
    parser.add_argument('--ins_encoder', type=str, default='clip', help="attribute loss type")
    parser.add_argument('--loss_type', type=str, default='mse', help="attribute loss type")
    parser.add_argument('--negative_num', type=int, default=10, help="negative_num")
    parser.add_argument('--num_of_MLP_layers', type=int, default=5, help="num_of_MLP_layers")
    parser.add_argument('--attribute_model_path', type=str, default='pre_trained_models/attribute_model_best.pth', help="attribute model path")
    parser.add_argument('--preferred_rate', type=float, default=1, help="preferred_rate")

    # ZSON
    parser.add_argument('--rgb_goal_dim', type=int, default=768, help="rgb goal dim")
    parser.add_argument('--random_seed', type=int, default=42, help='eval num')
    parser.add_argument('--using_map', type=bool, default=True, help='use map')
    parser.add_argument('--eval_pth', type=str, default="checkpoints/VTN_Local/attribute_model_1_44000.pth", help='directory to save eval results')

    # IL
    parser.add_argument('--path_to_train_traj_file', type=str, default='dataset/train_traj.json', help="path to traj file")
    parser.add_argument('--path_to_val_traj_file', type=str, default='dataset/val_traj.json', help="path to traj file")
    parser.add_argument('--il_lr', type=float, default=1e-5, help="il learning rate")
    parser.add_argument('--il_batch_size', type=int, default=1, help="il batch size")
    parser.add_argument('--max_seq_len', type=int, default=20, help="max seq len")
    parser.add_argument('--use_LLM_point_finder', type=bool, default=True, help='eval num')

    args = parser.parse_args()
    if args.use_gps_compass is False:
        args.gps_compass_embedding_dim = 0
    return args
