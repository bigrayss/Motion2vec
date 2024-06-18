import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from collections import namedtuple
from typing import Optional, Tuple
import numpy as np

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from model.MotionAGFormer import create_layers
from motion2vec.modelcfg import M2vPoseConfig, Motion2VecConfig

from motion2vec.ema_module import EMAModule, EMAModuleConfig


MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])
MaskSeed = namedtuple("MaskSeed", ["seed", "ids"])


class Pose2Vec(nn.Module):
    def __init__(self, args, skip_ema=False):
        super().__init__()
        self.args = args
        self.D2vPoseConfig = M2vPoseConfig
        self.cfg = Motion2VecConfig
        self.mask_emb = nn.Parameter(torch.FloatTensor(args.dim_feat).uniform_())
        self.joints_embed = nn.Linear(args.dim_in, args.dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, args.num_joints, args.dim_feat))
        self.norm = nn.LayerNorm(args.dim_feat)
        self.layers = create_layers(dim=args.dim_feat,
                                    n_layers=args.n_layers,
                                    mlp_ratio=args.mlp_ratio,
                                    act_layer=nn.GELU,
                                    attn_drop=args.attn_drop,
                                    drop_rate=args.drop,
                                    drop_path_rate=args.drop_path,
                                    num_heads=args.num_heads,
                                    use_layer_scale=args.use_layer_scale,
                                    qkv_bias=args.qkv_bias,
                                    qkv_scale=args.qkv_scale,
                                    layer_scale_init_value=args.layer_scale_init_value,
                                    use_adaptive_fusion=args.use_adaptive_fusion,
                                    hierarchical=args.hierarchical,
                                    use_temporal_similarity=args.use_temporal_similarity,
                                    temporal_connection_len=args.temporal_connection_len,
                                    use_tcn=args.use_tcn,
                                    graph_only=args.graph_only,
                                    neighbour_num=args.neighbour_num,
                                    n_frames=args.n_frames)
        self.dropout_input = nn.Dropout(self.cfg.dropout_input)
        self.ema = None
        self.average_top_k_layers = self.cfg.average_top_k_layers
        if not skip_ema:
            self.ema = self.make_ema_teacher(self.cfg.ema_decay)
        self.loss_beta = self.cfg.loss_beta
        self.loss_scale = self.cfg.loss_scale

    def compute_mask_indices(
            self,
            shape: Tuple[int, int, int],
            padding_mask: Optional[torch.Tensor],
            mask_prob: float,
            mask_length: int,
            mask_type: str = "static",
            mask_other: float = 0.0,
            min_masks: int = 0,
            no_overlap: bool = False,
            min_space: int = 0,
            require_same_masks: bool = True,
            mask_dropout: float = 0.0,
            add_masks: bool = False,
            seed: Optional[int] = None,
            epoch: Optional[int] = None,
            indices: Optional[torch.Tensor] = None,
            idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
            num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
    ):
        """
        Computes random mask spans for a given shape

        Args:
            shape: the shape for which to compute masks.
                should be of size 2 where first element is batch size and 2nd is timesteps
            padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
            mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
                number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
                however due to overlaps, the actual number will be smaller (unless no_overlap is True)
            mask_type: how to compute mask lengths
                static = fixed size
                uniform = sample from uniform distribution [mask_other, mask_length*2]
                normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
                poisson = sample from possion distribution with lambda = mask length
            min_masks: minimum number of masked spans
            no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
            min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
            require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
            mask_dropout: randomly dropout this percentage of masks in each example
        """

        bsz, frame_sz, jsz = shape

        if num_mask_ver == 1:
            all_num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * jsz / float(mask_length)
                + np.random.rand()
            )
            all_num_mask = max(min_masks, all_num_mask)

        all_mask_idcs = []
        for i in range(bsz):
            mask_idcs = []
            for j in range(frame_sz):
                if seed is not None and epoch is not None and indices is not None:
                    seed_i = int(hash((seed, epoch, indices[i + j].item())) % 1e6)
                else:
                    seed_i = None

                rng = np.random.default_rng(seed_i)

                if padding_mask is not None:
                    sz = jsz - padding_mask[j].long().sum().item()
                    assert sz >= 0, sz
                else:
                    sz = jsz

                if num_mask_ver == 1:
                    if padding_mask is not None:
                        num_mask = int(
                            # add a random number for probabilistic rounding
                            mask_prob * sz / float(mask_length)
                            + np.random.rand()
                        )
                        num_mask = max(min_masks, num_mask)
                    else:
                        num_mask = all_num_mask
                elif num_mask_ver == 2:
                    num_mask = int(
                        # add a random number for probabilistic rounding
                        mask_prob * sz / float(mask_length)
                        + rng.random()
                    )
                    num_mask = max(min_masks, num_mask)
                else:
                    raise ValueError()

                if mask_type == "static":
                    lengths = np.full(num_mask, mask_length)
                elif mask_type == "uniform":
                    lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
                elif mask_type == "normal":
                    lengths = rng.normal(mask_length, mask_other, size=num_mask)
                    lengths = [max(1, int(round(x))) for x in lengths]
                elif mask_type == "poisson":
                    lengths = rng.poisson(mask_length, size=num_mask)
                    lengths = [int(round(x)) for x in lengths]
                else:
                    raise Exception("unknown mask selection " + mask_type)

                if sum(lengths) == 0:
                    if mask_type == "static":
                        raise ValueError(f"this should never happens")
                    else:
                        lengths = [min(mask_length, sz - 1)]

                if no_overlap:
                    mask_idc = []

                    def arrange(s, e, length, keep_length):
                        span_start = rng.randint(s, e - length)
                        mask_idc.extend(span_start + j for j in range(length))

                        new_parts = []
                        if span_start - s - min_space >= keep_length:
                            new_parts.append((s, span_start - min_space + 1))
                        if e - span_start - length - min_space > keep_length:
                            new_parts.append((span_start + length + min_space, e))
                        return new_parts

                    parts = [(0, sz)]
                    min_length = min(lengths)
                    for length in sorted(lengths, reverse=True):
                        lens = np.fromiter(
                            (e - s if e - s >= length + min_space else 0 for s, e in parts),
                            np.int,
                        )
                        l_sum = np.sum(lens)
                        if l_sum == 0:
                            break
                        probs = lens / np.sum(lens)
                        c = rng.choice(len(parts), p=probs)
                        s, e = parts.pop(c)
                        parts.extend(arrange(s, e, length, min_length))
                    mask_idc = np.asarray(mask_idc)
                else:
                    if idc_select_ver == 1:
                        min_len = min(lengths)
                        if sz - min_len <= num_mask:
                            min_len = sz - num_mask - 1
                        mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
                    elif idc_select_ver == 2:
                        mask_idc = rng.choice(sz, num_mask, replace=False)
                    else:
                        raise ValueError()

                    mask_idc = np.asarray(
                        [
                            mask_idc[k] + offset
                            for k in range(len(mask_idc))
                            for offset in range(lengths[k])
                        ]
                    )

                mask_idc = np.unique(mask_idc[mask_idc < sz])
                if len(mask_idc) >= sz:
                    raise ValueError(
                        (
                            f"the entire sequence is masked. "
                            f"sz={sz}; mask_idc[mask_idc]; "
                            f"index={indices[j] if indices is not None else None}"
                        )
                    )
                mask_idcs.append(mask_idc)
            all_mask_idcs.append(mask_idcs)

        masks = []
        for mask_idcs in all_mask_idcs:
            mask = np.full((frame_sz, jsz), False)
            target_len = None
            if require_same_masks:
                if add_masks:
                    target_len = max([len(m) for m in mask_idcs])
                else:
                    target_len = min([len(m) for m in mask_idcs])

            for i, mask_idc in enumerate(mask_idcs):
                if target_len is not None and len(mask_idc) > target_len:
                    mask_idc = rng.choice(mask_idc, target_len, replace=False)

                mask[i, mask_idc] = True

                if target_len is not None and len(mask_idc) < target_len:
                    unmasked = np.flatnonzero(~mask[i])
                    to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
                    mask[i, to_mask] = True

                if mask_dropout > 0:
                    masked = np.flatnonzero(mask[i])
                    num_holes = np.rint(len(masked) * mask_dropout).astype(int)
                    to_drop = rng.choice(masked, num_holes, replace=False)
                    mask[i, to_drop] = False
            masks.append(mask)
        return np.array(masks)

    def make_maskinfo(self, x, mask, shape=None):
        if shape is None:
            B, T, J, D = x.shape
        else:
            B, T, J, D = shape

        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=2)
        ids_restore = ids_shuffle.argsort(dim=2).unsqueeze(-1).expand(-1, -1, -1, D)

        len_keep = J - mask[0, 0].sum()
        # if self.modality_cfg.keep_masked_pct > 0:
        #     len_keep += round((T - int(len_keep)) * self.modality_cfg.keep_masked_pct)

        ids_keep = ids_shuffle[:, :, :len_keep]

        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, -1, D)
            x_unmasked = torch.gather(x, dim=2, index=ids_keep)

        mask_info = MaskInfo(
            x_unmasked=x_unmasked,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )   # mask_info
        return mask_info  # 返回掩蔽的信息

    def gather_unmasked(self, x, mask_info):
        return torch.gather(
            x,
            dim=2,
            index=mask_info.ids_keep,
        )

    def is_xla_tensor(self, tensor):
        return torch.is_tensor(tensor) and tensor.device.type == "xla"

    def index_put(self, tensor, indices, value):
        if self.is_xla_tensor(tensor):
            for _ in range(indices.dim(), tensor.dim()):
                indices = indices.unsqueeze(-1)
            if indices.size(-1) < tensor.size(-1):
                indices = indices.expand_as(tensor)
            tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)  # value是
        else:
            tensor[indices] = value
        return tensor

    def denoise_feature(self, x, mask_info: MaskInfo):
        inp_drop = self.cfg.dropout_input
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        num_extra = self.D2vPoseConfig.num_extra_tokens

        if mask_info is not None:
            num_masked = mask_info.ids_restore.shape[2] - x.shape[2] + num_extra

            mask_tokens = x.new_empty(
                x.size(0),
                x.size(1),
                num_masked,
                x.size(-1),
            ).normal_(0, self.D2vPoseConfig.mask_noise_std)

            x_ = torch.cat([x[:, :, num_extra:], mask_tokens], dim=2)
            x = torch.gather(x_, dim=2, index=mask_info.ids_restore)

            # if D2vPoseConfig.decoder.add_positions_masked:
            #     assert self.fixed_positional_encoder is not None
            #     pos = self.fixed_positional_encoder(x, None)
            #     x = x + (pos * mask_info.mask.unsqueeze(-1))
        else:
            x = x[:, :, num_extra:]

        # if D2vPoseConfig.decoder.add_positions_all:
        #     assert self.fixed_positional_encoder is not None
        #     x = x + self.fixed_positional_encoder(x, None)

        return x

    def make_ema_teacher(self, ema_decay):
        ema_config = EMAModuleConfig()
        ema_config.ema_decay = ema_decay,
        ema_config.ema_fp32 = True,  # False
        ema_config.log_norms = self.cfg.log_norms,  # True
        ema_config.add_missing_params = False,

        model_copy = self.make_target_model()

        return EMAModule(
            model_copy,
            ema_config,
            copy_model=False,
        )

    def make_target_model(self):

        model_copy = Pose2Vec(args=self.args, skip_ema=True)

        if self.cfg.ema_encoder_only:  # True
            model_copy = model_copy.layers
            for p_s, p_t in zip(self.layers.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)

        model_copy.requires_grad_(False)

        return model_copy

    def make_targets(self, y, num_layers):

        with torch.no_grad():
            target_layer_results = y[-num_layers:]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
                ]
                permuted = True
            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]
            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]
            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]
            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        return y

    def forward(self, x, id=None):
        """
        :param source: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        """

        # pose feature extractor
        x = self.joints_embed(x)
        x = x + self.pos_embed  # position embedding on the spacial dimension

        pre_encoder_features = x.clone()

        if self.cfg.clone_batch > 1:
            x = x.repeat_interleave(self.cfg.clone_batch, 0)

        # x = self.dropout_input(x)

        # compute mask
        B, F, J, C = x.shape

        mask_prob = self.D2vPoseConfig.mask_prob
        mask_seed = MaskSeed(seed=self.cfg.seed, ids=id)
        mask = self.compute_mask_indices(
            shape=(B, F, J),
            padding_mask=None,
            mask_prob=mask_prob,  # 0.7
            mask_length=self.D2vPoseConfig.mask_length,  # 5
            min_masks=1,
            require_same_masks=True,
            mask_dropout=self.D2vPoseConfig.mask_dropout,
            add_masks=self.D2vPoseConfig.add_masks,  # False
            seed=mask_seed.seed if mask_seed is not None else None,
            epoch=None,
            indices=mask_seed.ids if mask_seed is not None else None,
        )

        mask = torch.from_numpy(mask).to(device=x.device)
        mask_info = self.make_maskinfo(x, mask)

        x = self.index_put(x, mask, self.mask_emb)

        # dx = self.denoise_feature(x, mask_info)

        # student context encoder
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # ema
        p = next(self.ema.model.parameters())
        device = x.device
        dtype = x.dtype
        ema_device = p.device
        ema_dtype = p.dtype

        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype
        if ema_device != device or ema_dtype != dtype:
            self.ema.model.to(dtype=dtype, device=device)
            pre_encoder_features = pre_encoder_features.to(dtype=dtype, device=device)
            ema_dtype = dtype

            def to_device(d):
                for k, p in enumerate(d):
                    if isinstance(d[k], nn.Sequential):
                        to_device(d[k])
                    else:
                        d[k] = p.to(device=device)
                    # print(device)

            to_device(self.ema.model)


        tm = self.ema.model

        # if torch.cuda.is_available():
        #     tm = torch.nn.DataParallel(tm)
        # tm.to('cuda')

        with torch.no_grad():
            tm.eval()
            if self.cfg.ema_encoder_only:
                ema_input = pre_encoder_features.to(dtype=ema_dtype, device=device)
                # ema_input = pre_encoder_features
                ema_block = tm

            # assert next(ema_block.parameters()).device == ema_input.device, "device error"

            y = []
            extra_tokens = self.D2vPoseConfig.num_extra_tokens
            for _, blk in enumerate(ema_block):
                ema_input = blk(ema_input)
                y.append(ema_input[:, extra_tokens:])

        y = self.make_targets(y, self.average_top_k_layers)

        if self.cfg.clone_batch > 1:
            y = y.repeat_interleave(self.cfg.clone_batch, 0)

        masked = mask.unsqueeze(-1)
        masked_b = mask.bool()
        y = y[masked_b]
        x = x[masked_b]

        sample_size = masked.sum().long()

        result = {
            "losses": {},
            "sample_size": sample_size,
        }

        if self.cfg.d2v_loss > 0:
            reg_loss = self.d2v_loss(x, y)
            result["losses"]["pose_regression"] = reg_loss * self.cfg.d2v_loss

        return result

    def d2v_loss(self, x, y):
        x = x.view(-1, x.size(-1)).float()
        y = y.view(-1, x.size(-1))

        if self.loss_beta == 0:
            loss = F.mse_loss(x, y, reduction="none")
        else:
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        reg_loss = loss * scale

        return reg_loss


if __name__ == '__main__':
    import argparse
    from torchsummary import summary
    from utils.tools import set_random_seed, get_config

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="configs/pretrain/Pose2Vec.yaml",
                            help="Path to the config file.")
        parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                            help='checkpoint directory')
        parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint',
                            help='new checkpoint directory')
        parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
        parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
        parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--eval-only', action='store_true')
        opts = parser.parse_args()
        return opts

    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)

    net = Pose2Vec(args)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net)
    net.to('cuda')
    x = torch.rand(1, 243, 17, 3).to('cuda')
    result = net(x)
    print(result["losses"]['pose_regression'].shape)
    # model = Pose2Vec(args).to('cuda')
    # summary(model, (243, 17, 3))