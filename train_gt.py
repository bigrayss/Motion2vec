import argparse
import numpy as np
# import wandb
import pkg_resources
from torch import optim
from tqdm import tqdm
import scipy.io as scio
import random

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader
from data.h36m_gt.h36m_dataset import Human36mDataset
from data.h36m_gt.load_dataset_h36m import Fusion
from data.h36m_gt.common import *

from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers
from utils.data import denormalize
from utils.utils_3dhp import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/MotionAGFormer-new.yaml",
                        help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default="checkpoint/finetune_0", type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, default="checkpoint", metavar='PATH',
                        help='new checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, default="best_epoch.pth.tr",
                        help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    # parser.add_argument('--use-wandb', action='store_true')
    # parser.add_argument('--wandb-name', default=None, type=str)
    # parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--P2V_reload', type=int, default=0)
    opts = parser.parse_args()
    return opts


def train_one_epoch(args, model, train_loader, optimizer, device, losses, reduce):
    model.train()
    for idx, data in enumerate(tqdm(train_loader)):
    # for x, y in tqdm(train_loader):
        if idx % 243 != reduce:
            continue
        # _, y, x, _, _, _, _, _ = data
        # dtype = y.dtype
        # batch_size = x.shape[0]
        # x, y = x.to(dtype=dtype, device=device), y.to(device)
        #
        # with torch.no_grad():
        #     if args.root_rel:
        #         y = y - y[..., 0:1, :]
        #     else:
        #         y[..., 2] = y[..., 2] - y[:, 0:1, 0:1, 2]  # Place the depth of first frame root to be 0
        #
        # pred = model(x)  # (N, T, 17, 3)

        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable('train',
                                                               [input_2D, gt_3D, batch_cam, scale, bb_box])

        batch_size = input_2D.size(0)
        y = gt_3D.clone().view(batch_size, -1, 17, 3)
        y[:, :, 0] = 0
        input_2D = input_2D.view(batch_size, -1, 17, 3).type(torch.cuda.FloatTensor)  # b c f j 1 -> b c f j

        pred = model(input_2D)  # b f j c
        pred = pred * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, pred.size(1), 17, 3)

        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)

        loss_total = loss_3d_pos + \
                    args.lambda_scale * loss_3d_scale + \
                    args.lambda_3d_velocity * loss_3d_velocity + \
                    args.lambda_lv * loss_lv + \
                    args.lambda_lg * loss_lg + \
                    args.lambda_a * loss_a + \
                    args.lambda_av * loss_av

        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()

# def evaluate(args, model, test_loader, datareader, device):
#     print("[INFO] Evaluation")
#     results_all = []
#     model.eval()
#     with torch.no_grad():
#         for x, y in tqdm(test_loader):
#             x, y = x.to(device), y.to(device)

#             if args.flip:
#                 batch_input_flip = flip_data(x)
#                 predicted_3d_pos_1 = model(x)
#                 predicted_3d_pos_flip = model(batch_input_flip)
#                 predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
#                 predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
#             else:
#                 predicted_3d_pos = model(x)
#             if args.root_rel:
#                 predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
#             else:
#                 y[:, 0, 0, 2] = 0

#             results_all.append(predicted_3d_pos.cpu().numpy())

#     results_all = np.concatenate(results_all)
#     results_all = datareader.denormalize(results_all)
#     _, split_id_test = datareader.get_split_id()
#     actions = np.array(datareader.dt_dataset['test']['action'])
#     factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
#     gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
#     sources = np.array(datareader.dt_dataset['test']['source'])

#     num_test_frames = len(actions)
#     frames = np.array(range(num_test_frames))
#     action_clips = actions[split_id_test]
#     factor_clips = factors[split_id_test]
#     source_clips = sources[split_id_test]
#     frame_clips = frames[split_id_test]
#     gt_clips = gts[split_id_test]
#     if args.add_velocity:
#         action_clips = action_clips[:, :-1]
#         factor_clips = factor_clips[:, :-1]
#         frame_clips = frame_clips[:, :-1]
#         gt_clips = gt_clips[:, :-1]

#     assert len(results_all) == len(action_clips)

#     e1_all = np.zeros(num_test_frames)
#     jpe_all = np.zeros((num_test_frames, args.num_joints))
#     e2_all = np.zeros(num_test_frames)
#     acc_err_all = np.zeros(num_test_frames - 2)
#     oc = np.zeros(num_test_frames)
#     results = {}
#     results_procrustes = {}
#     results_joints = [{} for _ in range(args.num_joints)]
#     results_accelaration = {}
#     action_names = sorted(set(datareader.dt_dataset['test']['action']))
#     for action in action_names:
#         results[action] = []
#         results_procrustes[action] = []
#         results_accelaration[action] = []
#         for joint_idx in range(args.num_joints):
#             results_joints[joint_idx][action] = []

#     block_list = ['s_09_act_05_subact_02',
#                   's_09_act_10_subact_02',
#                   's_09_act_13_subact_01']
#     for idx in range(len(action_clips)):
#         source = source_clips[idx][0][:-6]
#         if source in block_list:
#             continue
#         frame_list = frame_clips[idx]
#         action = action_clips[idx][0]
#         factor = factor_clips[idx][:, None, None]
#         gt = gt_clips[idx]
#         pred = results_all[idx]
#         pred *= factor

#         # Root-relative Errors
#         pred = pred - pred[:, 0:1, :]
#         gt = gt - gt[:, 0:1, :]
#         err1 = calculate_mpjpe(pred, gt)
#         jpe = calculate_jpe(pred, gt)
#         for joint_idx in range(args.num_joints):
#             jpe_all[frame_list, joint_idx] += jpe[:, joint_idx]
#         acc_err = calculate_acc_err(pred, gt)
#         acc_err_all[frame_list[:-2]] += acc_err
#         e1_all[frame_list] += err1
#         err2 = calculate_p_mpjpe(pred, gt)
#         e2_all[frame_list] += err2
#         oc[frame_list] += 1
#     for idx in range(num_test_frames):
#         if e1_all[idx] > 0:
#             err1 = e1_all[idx] / oc[idx]
#             err2 = e2_all[idx] / oc[idx]
#             action = actions[idx]
#             results_procrustes[action].append(err2)
#             acc_err = acc_err_all[idx] / oc[idx]
#             results[action].append(err1)
#             results_accelaration[action].append(acc_err)
#             for joint_idx in range(args.num_joints):
#                 jpe = jpe_all[idx, joint_idx] / oc[idx]
#                 results_joints[joint_idx][action].append(jpe)
#     final_result_procrustes = []
#     final_result_joints = [[] for _ in range(args.num_joints)]
#     final_result_acceleration = []
#     final_result = []

#     for action in action_names:
#         final_result.append(np.mean(results[action]))
#         final_result_procrustes.append(np.mean(results_procrustes[action]))
#         final_result_acceleration.append(np.mean(results_accelaration[action]))
#         for joint_idx in range(args.num_joints):
#             final_result_joints[joint_idx].append(np.mean(results_joints[joint_idx][action]))

#     joint_errors = []
#     for joint_idx in range(args.num_joints):
#         joint_errors.append(
#             np.mean(np.array(final_result_joints[joint_idx]))
#         )
#     joint_errors = np.array(joint_errors)
#     e1 = np.mean(np.array(final_result))
#     assert round(e1, 4) == round(np.mean(joint_errors), 4), f"MPJPE {e1:.4f} is not equal to mean of joint errors {np.mean(joint_errors):.4f}"
#     acceleration_error = np.mean(np.array(final_result_acceleration))
#     e2 = np.mean(np.array(final_result_procrustes))
#     print('Protocol #1 Error (MPJPE):', e1, 'mm')
#     print('Acceleration error:', acceleration_error, 'mm/s^2')
#     print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
#     print('----------')
#     return e1, e2, joint_errors, acceleration_error


def input_augmentation(input_2D, model, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1]
    input_2D_non_flip = input_2D[:, 0]
    output_3D_flip = model(input_2D_flip)
    output_3D_flip[..., 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
    output_3D_non_flip = model(input_2D_non_flip)
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip

    return input_2D, output_3D


# def evaluate(model, test_loader, n_frames):
#     print("[INFO] Evaluation")
#     model.eval()
#     joints_left = [5, 6, 7, 11, 12, 13]
#     joints_right = [2, 3, 4, 8, 9, 10]
#
#     data_inference = {}
#     error_sum_test = AccumLoss()
#
#     for idx, data in enumerate(tqdm(test_loader)):
#         if idx % 175 != 0:
#             continue
#         # batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
#
#         batch_cam, gt_3D, input_2D, action, seq, scale, bb_box, cam_ind = data
#         [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable('test', [input_2D, gt_3D, batch_cam, scale, bb_box])
#
#         N = input_2D.size(0)
#
#         out_target = gt_3D.clone().view(N, -1, 17, 3)
#         out_target[:, :, 14] = 0
#         gt_3D = gt_3D.view(N, -1, 17, 3).type(torch.cuda.FloatTensor)
#
#         input_2D, output_3D = input_augmentation(input_2D, model, joints_left, joints_right)
#
#         output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
#         pad = (n_frames - 1) // 2
#         pred_out = output_3D[:, pad].unsqueeze(1)
#
#         pred_out[..., 14, :] = 0
#         pred_out = denormalize(pred_out, seq)
#
#         pred_out = pred_out - pred_out[..., 14:15, :] # Root-relative prediction
#
#         inference_out = pred_out + out_target[..., 14:15, :] # final inference (for PCK and AUC) is not root relative
#
#         out_target = out_target - out_target[..., 14:15, :] # Root-relative prediction
#
#         joint_error_test = mpjpe_cal(pred_out, out_target).item()
#
#         for seq_cnt in range(len(seq)):
#             seq_name = seq[seq_cnt]
#             if seq_name in data_inference:
#                 data_inference[seq_name] = np.concatenate(
#                     (data_inference[seq_name], inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
#             else:
#                 data_inference[seq_name] = inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy()
#
#         error_sum_test.update(joint_error_test * N, N)
#
#     for seq_name in data_inference.keys():
#         data_inference[seq_name] = data_inference[seq_name][:, :, None, :]
#
#     print(f'Protocol #1 Error (MPJPE): {error_sum_test.avg:.2f} mm')
#
#     return error_sum_test.avg, data_inference


def evaluate(model, test_loader, n_frames, reduce):
     print("[INFO] Evaluation")
     model.eval()
     actions = define_actions('*')
     action_error_sum = define_error_list(actions)

     joints_left = [4, 5, 6, 11, 12, 13]
     joints_right = [1, 2, 3, 14, 15, 16]

     for i, data in enumerate(tqdm(test_loader, 0)):
         if i % 243 != reduce:
            continue
         batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
         [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable('test',
                                                                    [input_2D, gt_3D, batch_cam, scale, bb_box])

         N = input_2D.size(0)

         out_target = gt_3D.clone().view(N, -1, 17, 3)
         out_target[:, :, 0] = 0

         input_2D, output_3D = input_augmentation(input_2D, model, joints_left, joints_right)

         output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
         pad = (n_frames - 1) // 2
         pred_out = output_3D[:, pad].unsqueeze(1)
         pred_out[:, :, 0, :] = 0

         action_error_sum = test_calculation(pred_out, out_target, action, action_error_sum)

     p1, p2 = print_error(action_error_sum, 0)
     return p1, p2


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe):  # , wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        # 'wandb_id': wandb_id,
    }, checkpoint_path)

def save_data_inference(path, data_inference, latest):
    if latest:
        mat_path = os.path.join(path, 'inference_data.mat')
    else:
        mat_path = os.path.join(path, 'inference_data_best.mat')
    scio.savemat(mat_path, data_inference)



def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)

    # train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    # test_dataset = MotionDataset3D(args, args.subset_list, 'test')

    dataset_path = args.root_path + args.data_file_3d
    dataset = Human36mDataset(dataset_path, args)

    train_dataset = Fusion(opt=args, train=True, dataset=dataset, root_path=args.root_path)
    test_dataset = Fusion(opt=args, train=False, dataset=dataset, root_path=args.root_path)

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': (opts.num_cpus - 1) // 3,
        'persistent_workers': True
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(opts.num_cpus), pin_memory=True)
    #
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=int(opts.num_cpus), pin_memory=True)

    # datareader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
    #                             data_stride_train=args.n_frames // 3, data_stride_test=args.n_frames,
    #                             dt_root='data/motion3d', dt_file=args.dt_file)  # Used for H36m evaluation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params:,}")

    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr,
                            weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float('inf')  # Used for storing the best model
    # wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()

    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            if opts.P2V_reload == 1:
                model_dict = model.state_dict()
                pre_dict = torch.load(checkpoint_path)
                state_dict = {k: v for k, v in pre_dict['model'].items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                model.load_state_dict(checkpoint['model'], strict=True)

            if opts.resume:
                if not opts.restart:
                    lr = checkpoint['lr']
                    epoch_start = checkpoint['epoch']
                    optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                # if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                #     wandb_id = checkpoint['wandb_id']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    # if not opts.eval_only:
    #     if opts.resume:
    #         if opts.use_wandb:
    #             wandb.init(id=wandb_id,
    #                     project='MotionMetaFormer',
    #                     resume="must",
    #                     settings=wandb.Settings(start_method='fork'))
    #     else:
    #         print(f"Run ID: {wandb_id}")
    #         if opts.use_wandb:
    #             wandb.init(id=wandb_id,
    #                     name=opts.wandb_name,
    #                     project='MotionMetaFormer',
    #                     settings=wandb.Settings(start_method='fork'))
    #             wandb.config.update({"run_id": wandb_id})
    #             wandb.config.update(args)
    #             installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
    #             wandb.config.update({'installed_packages': installed_packages})

    checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    for epoch in range(epoch_start, args.epochs):
        reduce = 5  # random.randint(0, 10)
        if opts.eval_only:
            with torch.no_grad():
                evaluate(model, test_loader, args.n_frames, reduce)
                exit()
                # evaluate(args, model, test_loader, datareader, device)

        print(f"[INFO] epoch {epoch}")
        loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, device, losses, reduce)

        # mpjpe, p_mpjpe, joints_error, acceleration_error = evaluate(args, model, test_loader, datareader, device)
        
        # with torch.no_grad():
        #     mpjpe, data_inference = evaluate(model, test_loader, args.n_frames)
        #
        # if mpjpe < min_mpjpe:
        #     min_mpjpe = mpjpe
        #     save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
        #     save_data_inference(opts.new_checkpoint, data_inference, latest=False)
        # save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
        # save_data_inference(opts.new_checkpoint, data_inference, latest=True)

        with torch.no_grad():
            mpjpe, p_mpjpe = evaluate(model, test_loader, args.n_frames, reduce)

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe)  # , wandb_id)
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, mpjpe)  # , wandb_id)

        # if opts.use_wandb:
        #     wandb.log({
        #         'lr': lr,
        #         'train/loss_3d_pose': losses['3d_pose'].avg,
        #         'train/loss_3d_scale': losses['3d_scale'].avg,
        #         'train/loss_3d_velocity': losses['3d_velocity'].avg,
        #         'train/loss_2d_proj': losses['2d_proj'].avg,
        #         'train/loss_lg': losses['lg'].avg,
        #         'train/loss_lv': losses['lv'].avg,
        #         'train/loss_angle': losses['angle'].avg,
        #         'train/angle_velocity': losses['angle_velocity'].avg,
        #         'train/total': losses['total'].avg,
        #         'eval/mpjpe': mpjpe,
        #         'eval/min_mpjpe': min_mpjpe,
        #     }, step=epoch + 1)

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)

    # if opts.use_wandb:
    #     artifact = wandb.Artifact(f'model', type='model')
    #     artifact.add_file(checkpoint_path_latest)
    #     artifact.add_file(checkpoint_path_best)
    #     wandb.log_artifact(artifact)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)
    print(f"[INFO] {opts.checkpoint}/{opts.checkpoint_file}")
    train(args, opts)


if __name__ == '__main__':
    main()
