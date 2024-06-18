"""
@Time : 2023/10/14 下午8:57
@Author : Ray
@Email : 1206953809@qq.com
@File : modelcfg.py
@Purpose
"""

# from omegaconf import II
from dataclasses import field


class M2vPoseConfig:
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0

    num_extra_tokens: int = 0
    init_extra_token_zero: bool = True

    mask_noise_std: float = 0.01
    mask_prob: float = 0.6
    inverse_mask: bool = False
    mask_prob_adjust: float = 0
    keep_masked_pct: float = 0
    mask_length: int = 5
    add_masks: bool = False
    remove_masks: bool = False
    mask_dropout: float = 0.0
    encoder_zero_mask: bool = True
    mask_channel_prob: float = 0.0
    mask_channel_length: int = 64


class Motion2VecConfig:

    # loss_beta: float = field(
    #     default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    # )
    # loss_scale: Optional[float] = field(
    #     default=None,
    #     metadata={
    #         "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
    #     },
    # )

    loss_beta: float = 0
    loss_scale: float = None

    # average_top_k_layers: int = field(
    #     default=8, metadata={"help": "how many layers to average"}
    # )

    average_top_k_layers: int = 6  # 1-8 2-4 3-8 4-4

    clone_batch: int = 3  # 1-3 2-3 3-1 4-3

    dropout_input: float = 0.0

    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool = True
    log_norms: bool = True
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    ema_encoder_only: bool = field(
        default=True,
        metadata={
            "help": "whether to momentum update only the shared transformer encoder"
        },
    )

    seed: int = 0 # II("common.seed")

    skip_ema: bool = False

    d2v_loss: float = 1
