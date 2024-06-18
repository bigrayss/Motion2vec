import argparse
import os

import numpy as np
import pkg_resources
import torch
import wandb
from torch import optim
from tqdm import tqdm

from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader

from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain/Pose2Vec.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, default="configs/pretrain/Pose2Vec.yaml", metavar='PATH', help='new checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, default="best_epoch.pth.tr", help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts


def train_one_epoch(model, train_loader, optimizer, device, losses):
    model.train()
    for x, _ in tqdm(train_loader):
        batch_size = x.shape[0]
        x = x.to(device)

        result = model(x)

        loss = result["losses"]["pose_regression"].mean()

        optimizer.zero_grad()

        losses["pose_regression"].update(loss.item(), batch_size)

        loss.backward()
        optimizer.step()


def evaluate(args, model, test_loader, datareader, device):
    print("[INFO] Evaluation")
    results_all = []
    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(test_loader):
            x = x.to(device)

            if args.flip:
                batch_input_flip = flip_data(x)
                result_1 = model(x)
                pose_loss_1 = result_1["losses"]["pose_regression"].mean()
                result_2 = model(batch_input_flip)
                pose_loss_2 = result_2["losses"]["pose_regression"].mean()
                pose_loss = (pose_loss_1 + pose_loss_2) / 2
            else:
                result = model(x)
                pose_loss = result["losses"]["pose_regression"].mean()

            results_all.append(pose_loss.cpu().numpy())

    p2v_loss = np.mean(results_all)

    print('L2 Error:', p2v_loss)
    return p2v_loss


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        }, checkpoint_path)


def train(args, opts):
    print_args(args)

    create_directory_if_not_exists(opts.new_checkpoint)

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')

    new_dataset = train_dataset.__add__(test_dataset)

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': (opts.num_cpus - 1) // 3,
        'persistent_workers': True
    }

    # train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    train_loader = DataLoader(new_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)
    datareader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                                data_stride_train=args.n_frames // 3, data_stride_test=args.n_frames,
                                dt_root='data/motion3d', dt_file=args.dt_file)  # Used for H36m evaluation

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

    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)

            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    for epoch in range(epoch_start, args.epochs):
        print(f"[INFO] epoch {epoch}")

        loss_names = ['pose_regression']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(model, train_loader, optimizer, device, losses)

        mpjpe = evaluate(args, model, test_loader, datareader, device)

        checkpoint_inter = opts.new_checkpoint + '/epoch%d.pth.tr' % (epoch)
        save_checkpoint(checkpoint_inter, epoch, lr, optimizer, model, mpjpe)

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe)
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe)

        print('learning rate: ', lr)
        lr = decay_lr_exponentially(lr, lr_decay, optimizer)


if __name__ == '__main__':
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)

    from motion2vec.modelcfg import M2vPoseConfig, Motion2VecConfig

    print(f"[INFO] mask_prob: {M2vPoseConfig.mask_prob}")
    print(f"[INFO] mask_length: {M2vPoseConfig.mask_length}")
    print(f"[INFO] clone_batch: {Motion2VecConfig.clone_batch}")
    print(f"[INFO] average_top_k_layers: {Motion2VecConfig.average_top_k_layers}")

    train(args, opts)