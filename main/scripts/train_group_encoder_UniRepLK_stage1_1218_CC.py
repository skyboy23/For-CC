'''
stage1
/root/autodl-tmp/code/43.algonauts23-CC/config/phase1_head_only_unireplk.yaml
--out_dir
/root/autodl-tmp/code/43.algonauts23-CC/scripts/results
--workers
4

2025.03.27 整理交付给CC

'''

import warnings
warnings.filterwarnings("ignore")

import fnmatch
import json
import logging
import math
import shutil
import sys
import time
from argparse import Namespace
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, load_from_disk
from hf_argparser import HfArg, HfArgumentParser
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import r2_score
from timm.data.random_erasing import RandomErasing
from timm.utils import AverageMeter, random_seed
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms as T

import wandb
# from .. import algonauts23
from algonauts23 import ALGONAUTS_DATA_DIR, SUBS
from algonauts23.features_unireplk import FeatureExtractor
from algonauts23.models import create_encoder, list_encoders
from algonauts23.models.trunks import DataConfig
from algonauts23.slug import random_slug
from algonauts23.space import AlgonautsSpace
from algonauts23.utils import get_sha, seed_hash, setup_logging
from algonauts23.visualize import (
    Visualizer,
    plot_maps,
    plot_pred_triplets,
    plot_roi_scores,
)

import cortex

# UniRepLK
import mmcv
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import (load_checkpoint)
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from mmdet.apis import init_random_seed, set_random_seed
import os.path as osp
import torch.distributed as dist

import mmdet_custom
import mmcv_custom
from timm.data import resolve_model_data_config, str_to_interp_mode

# UniRepLK

np.set_printoptions(precision=3)
plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 150

Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

PROJECT = "algonauts23-group-encoder"

LOG_INTERVAL = 10
NUM_EXAMPLES = 12
LOG_GRAD_INTERVAL = 500


@dataclass
class Args:
    # Architecture
    trunk: str = HfArg(
        default="eva02_base_patch14_224.mim_in22k", help="timm pretrained model name"
        # default="/home/mashuxiao/code/43.algonauts23-main_BlobGPT/eva02clip/model.safetensors", help="local file"
    )
    layers: List[str] = HfArg(
        default_factory=lambda: ["blocks.[02468]", "blocks.10"],
        help="list of layer names to extract",
    )
    encoder: str = HfArg(
        default="grouplin", help=f"encoder model ({', '.join(list_encoders())})"
    )
    # Paths
    out_dir: str = HfArg(default="results", help="path to root output directory")
    data_dir: str = HfArg(
        default=str(ALGONAUTS_DATA_DIR), help="path to algonauts root data directory"
    )
    name: Optional[str] = HfArg(default=None, help="experiment name")
    prefix: Optional[str] = HfArg(default=None, help="experiment name prefix")
    desc: Optional[str] = HfArg(default=None, help="description to attach to run")
    # Extra architecture settings
    hidden_dim: int = HfArg(aliases=["--hd"], default=1024, help="hidden dimension")
    proj_dim: int = HfArg(
        aliases=["--pd"],
        default=2048,
        help="final projection dimension, must match embed_state if provided",
    )
    embed_state: Optional[str] = HfArg(
        default=None,
        help="torch state containing weight and bias for final embedding to target "
             "dimension",
    )
    # Regularization and augmentation
    dropout: float = HfArg(default=0.8, help="feature dropout rate")
    norm: bool = HfArg(default=True, help="batch normalize encoder latent features")
    crop_scale: float = HfArg(default=0.8, help="random resize crop min scale")
    jitter_prob: float = HfArg(default=0.0, help="color jitter probability")
    gray_prob: float = HfArg(default=0.0, help="gray scale probability")
    blur_prob: float = HfArg(default=0.0, help="gaussian blur probability")
    hflip_prob: float = HfArg(default=0.0, help="horizontal flip probability")
    re_prob: float = HfArg(default=0.0, help="random erase probability")
    trunk_dropout: float = HfArg(
        aliases=["--td"], default=0.0, help="trunk attn/proj dropout rate"
    )
    trunk_drop_path: float = HfArg(
        aliases=["--tdp"], default=0.0, help="trunk drop path rate"
    )
    trunk_drop_patch: float = HfArg(default=0.0, help="trunk drop patch rate")
    freeze_patterns: Optional[List[str]] = HfArg(
        default=None, help="list of patterns for parameters to freeze"
    )
    unfreeze_patterns: Optional[List[str]] = HfArg(
        default=None,
        help="list of patterns for parameters to unfreeze (applied after freeze)",
    )
    # Optimization
    fine_tune: bool = HfArg(aliases=["--ft"], default=False, help="fine-tune the trunk")
    epochs: int = HfArg(default=10, help="number of epochs")
    batch_size: int = HfArg(aliases=["--bs"], default=128, help="batch size")
    lr: float = HfArg(default=1e-3, help="learning rate")
    decay_lr: bool = HfArg(default=True, help="decay learning rate")
    warmup_fraction: float = HfArg(
        default=0.05, help="number of warmup steps as a fraction of total"
    )
    min_lr_fraction: float = HfArg(
        default=0.05, help="minimum lr as a fraction of max lr"
    )
    weight_decay: float = HfArg(aliases=["--wd"], default=0.8, help="weight decay")
    beta1: float = HfArg(default=0.9, help="AdamW beta1")
    beta2: float = HfArg(default=0.99, help="AdamW beta2")
    grad_accum_steps: int = HfArg(
        aliases=["--accum"], default=1, help="number of gradient accumulation steps"
    )
    clip_grad: Optional[float] = HfArg(default=None, help="gradient norm clipping")
    # Logistics
    checkpoint: Optional[str] = HfArg(
        aliases=["--ckpt"], default=None, help="checkpoint to load"
    )
    cuda: bool = HfArg(default=True, help="use cuda")
    amp: bool = HfArg(default=False, help="use AMP")
    workers: int = HfArg(aliases=["-j"], default=4, help="data loading workers")
    overwrite: bool = HfArg(default=False, help="overwrite pre-existing results")
    figures: bool = HfArg(default=True, help="generate validation figures")
    wandb: bool = HfArg(default=False, help="log to wandb")
    sweep: bool = HfArg(default=False, help="whether we're in a wandb sweep")
    debug: bool = HfArg(default=False, help="quick debug mode")
    seed: int = HfArg(default=42, help="random seed")


def create_unireplk_train() -> Tuple[torch.nn.Module, DataConfig]:
    # args = parse_args()

    config_file_path = "/home/mashuxiao/code/48.UniRepLKNet-main/detection/configs/coco" \
                       "/casc_mask_rcnn_unireplknet_xl_in22k_fpn_3x_coco_MA.py"
    cfg = Config.fromfile(config_file_path)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)
    #
    # if args.auto_scale_lr:
    #     if 'auto_scale_lr' in cfg and \
    #             'enable' in cfg.auto_scale_lr and \
    #             'base_batch_size' in cfg.auto_scale_lr:
    #         cfg.auto_scale_lr.enable = True
    #     else:
    #         warnings.warn('Can not find "auto_scale_lr" or '
    #                       '"auto_scale_lr.enable" or '
    #                       '"auto_scale_lr.base_batch_size" in your'
    #                       ' configuration file. Please update all the '
    #                       'configuration files to mmdet >= 2.24.1.')

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # # work_dir is determined in this priority: CLI > segment in file > filename
    # if args.work_dir is not None:
    #     # update configs according to CLI args if args.work_dir is not None
    #     cfg.work_dir = args.work_dir
    # elif cfg.get('work_dir', None) is None:
    #     # use config filename as default work_dir if cfg.work_dir is None
    #     cfg.work_dir = osp.join('./work_dirs',
    #                             osp.splitext(osp.basename(args.config))[0])

    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_file_path))[0])

    resume_from = "/home/mashuxiao/code/48.UniRepLKNet-main/detection/ckpt/casc_mask_rcnn_unireplknet_xl_in22k_3x_coco_ap56.39.pth"

    if resume_from is not None:
        cfg.resume_from = resume_from
    cfg.auto_resume = False
    # if args.gpus is not None:
    #     cfg.gpu_ids = range(1)
    #     warnings.warn('`--gpus` is deprecated because we only support '
    #                   'single GPU mode in non-distributed training. '
    #                   'Use `gpus=1` now.')
    # if args.gpu_ids is not None:
    #     cfg.gpu_ids = args.gpu_ids[0:1]
    #     warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
    #                   'Because we only support single GPU mode in '
    #                   'non-distributed training. Use the first GPU '
    #                   'in `gpu_ids` now.')
    # if args.gpus is None and args.gpu_ids is None:
    #     cfg.gpu_ids = [args.gpu_id]
    cfg.gpu_ids = [0]

    # # init distributed env first, since logger depends on the dist info.
    # if args.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)
    #     # re-set gpu_ids with distributed training mode
    #     _, world_size = get_dist_info()
    #     cfg.gpu_ids = range(world_size)
    distributed = False

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config_file_path)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    # seed = seed + dist.get_rank() if args.diff_seed else seed
    args.deterministic = False
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(config_file_path)

    model = build_detector(cfg.model,
                           train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    # model.init_weights()

    data_config = resolve_model_data_config(model)

    img_size = data_config["input_size"][1]
    interp_mode = str_to_interp_mode(data_config["interpolation"])
    cfg = DataConfig(
        img_size=img_size,
        mean=data_config["mean"],
        std=data_config["std"],
        interp_mode=interp_mode,
    )
    return model, cfg


def create_unireplk_eval() -> Tuple[torch.nn.Module, DataConfig]:
    # UniRepLK
    UniRepLK_cfg = Config.fromfile("/root/autodl-tmp/code/48.UniRepLKNet-main/detection/configs/coco/"
                                   "casc_mask_rcnn_unireplknet_xl_in22k_fpn_3x_coco.py")

    UniRepLK_cfg.gpu_ids = [0]

    # build the UniRepLK model and load checkpoint
    model = build_detector(UniRepLK_cfg.model, test_cfg=UniRepLK_cfg.get('test_cfg'),
                           train_cfg=UniRepLK_cfg.get('train_cfg'))

    data_config = resolve_model_data_config(model)

    img_size = data_config["input_size"][1]
    interp_mode = str_to_interp_mode(data_config["interpolation"])
    cfg = DataConfig(
        img_size=img_size,
        mean=data_config["mean"],
        std=data_config["std"],
        interp_mode=interp_mode,
    )
    return model, cfg


def main(args: Args):
    start_time = time.monotonic()
    random_seed(args.seed)

    commit_sha = get_sha()
    if args.name is not None:
        name = args.name
    else:
        name = datetime.now().strftime("%y%m%d%H%M%S")
        if args.prefix:
            name = name + "-" + args.prefix
        name_seed = seed_hash(commit_sha, json.dumps(args.__dict__))
        name = name + "-" + random_slug(seed=name_seed)
    out_dir = Path(args.out_dir) / PROJECT
    if args.sweep:
        out_dir = out_dir / "sweeps"
    out_dir = out_dir / name

    overwritten = False
    if out_dir.exists():
        if args.overwrite:
            overwritten = True
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Output directory {out_dir} already exists")
    out_dir.mkdir(parents=True)
    setup_logging(out_dir)

    if args.wandb and not args.sweep:
        wandb.init(project=PROJECT, name=name, config=args.__dict__)

    logging.info("Starting training: %s/%s", PROJECT, name)
    logging.info("Args:\n%s", yaml.safe_dump(args.__dict__, sort_keys=False))
    logging.info(commit_sha)

    logging.info("Writing to %s", out_dir)
    if overwritten:
        logging.warning("Overwriting previous results")
    with (out_dir / "args.yaml").open("w") as f:
        yaml.safe_dump(args.__dict__, f, sort_keys=False)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logging.info("Running on: %s", device)

    if args.amp:
        autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)
        scaler = GradScaler()
        logging.info("Running in mixed precision with native PyTorch AMP")
    else:
        autocast = suppress
        scaler = None

    logging.info("Constructing feature extractor: %s, %s", args.trunk, args.layers)

    # if not args.fine_tune:
    #     trunk, data_config = create_unireplk_eval()
    # else:
    #     trunk, data_config = create_unireplk_train()

    trunk, data_config = create_unireplk_eval()
    trunk_ckpt = load_checkpoint(trunk, "/root/autodl-tmp/code/48.UniRepLKNet-main/detection/ckpt"
                                        "/casc_mask_rcnn_unireplknet_xl_in22k_3x_coco_ap56.39.pth", map_location='cpu')

    # Freeze/unfreeze trunk  # 阶段一 默认是关闭trunk的梯度更新
    for p in trunk.parameters():
        p.requires_grad_(args.fine_tune)
    trunk = trunk.to(device)
    if not args.fine_tune:
        trunk.eval()


    extractor = FeatureExtractor(trunk, args.layers, detach=False)
    feature_shapes = get_feature_shapes(extractor, data_config.img_size,
                                        device)  # [(192, 56, 56), (384, 28, 28), (768, 14, 14), (1536, 7, 7)]
    # feature_shapes = [(256, 3136), (512, 784), (1024, 196), (2048, 49)]
    logging.info(
        "Extracting layers:\n%s",
        {layer: shape for layer, shape in zip(extractor.layers, feature_shapes)},
    )

    logging.info("Constructing encoder: %s", args.encoder)
    encoder = create_encoder(
        args.encoder,
        feature_shapes=feature_shapes,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        norm=args.norm,
        embed_state=args.embed_state,
    )
    encoder = encoder.to(device)
    logging.info("Encoder:\n%s", encoder)

    named_params = list(encoder.named_parameters())
    if args.fine_tune:
        named_params.extend(trunk.named_parameters(prefix="trunk"))

    if args.freeze_patterns:
        logging.info("Freezing parameters matching patterns: %s", args.freeze_patterns)
        frozen = set_requires_grad(
            named_params, args.freeze_patterns, requires_grad=False
        )
        logging.info("Frozen parameters[:10]:\n%s", "\n".join(frozen[:10]))

    if args.unfreeze_patterns:
        logging.info(
            "Unfreezing parameters matching patterns: %s", args.unfreeze_patterns
        )
        unfrozen = set_requires_grad(
            named_params, args.unfreeze_patterns, requires_grad=True
        )
        logging.info("Unfrozen parameters[:10]:\n%s", "\n".join(unfrozen[:10]))

    logging.info(
        "Trunk params (trainable): %.0fM (%.0fM)",
        sum(p.numel() for p in trunk.parameters()) / 1e6,
        sum(p.numel() for p in trunk.parameters() if p.requires_grad) / 1e6,
    )
    logging.info(
        "Encoder params (trainable): %.0fM (%.0fM)",
        sum(p.numel() for p in encoder.parameters()) / 1e6,
        sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6,
    )

    logging.info("Loading dataset")
    dsets = {}
    for split in ["train", "val", "testval", "test"]:
        dsets[split] = ds = load_dataset(args, split=split)
        logging.info("%s samples: %d", split.capitalize(), len(ds))

    train_transform = create_transform(args, data_config, train=True)
    val_transform = create_transform(args, data_config, train=False)
    logging.info("Train transform:\n%s", train_transform)
    logging.info("Val transform:\n%s", val_transform)

    loaders = {}
    for split, ds in dsets.items():
        transform = train_transform if split == "train" else val_transform
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.workers,
            pin_memory=use_cuda,
            collate_fn=make_collate(transform),
        )

    params = list(encoder.parameters())
    if args.fine_tune:
        params.extend(trunk.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)

    # HACK: subj01 has complete data so it can work as a group visualizer

    cortex.download_subject(subject_id="fsaverage")
    visualizer = Visualizer(
        "subj01", root=Path(args.data_dir) / "algonauts_2023_challenge_data"
    )
    spaces = {
        sub: AlgonautsSpace(
            sub, root=Path(args.data_dir) / "algonauts_2023_challenge_data"
        )
        for sub in SUBS
    }

    if args.checkpoint:
        logging.info("Loading checkpoint: %s", args.checkpoint)
        load_checkpoint_Blob(
            args.checkpoint, trunk, encoder, device, load_trunk=args.fine_tune
        )

    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = -1
    epoch_steps = math.ceil(len(loaders["train"]) / args.grad_accum_steps)

    for epoch in range(start_epoch, args.epochs):
        logging.info("Starting epoch %d", epoch)

        train(
            args=args,
            epoch=epoch,
            extractor=extractor,
            encoder=encoder,
            train_loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            visualizer=visualizer,
            device=device,
            autocast=autocast,
            scaler=scaler,
            out_dir=out_dir,
        )

        metric = validate(
            args=args,
            epoch=epoch,
            step=(epoch + 1) * epoch_steps,
            extractor=extractor,
            encoder=encoder,
            val_loader=loaders["val"],
            criterion=criterion,
            spaces=spaces,
            visualizer=visualizer,
            device=device,
            out_dir=out_dir,
        )

        save_checkpoint(
            args=args,
            epoch=epoch,
            trunk=trunk,
            encoder=encoder,
            metric=metric,
            is_best=(metric > best_metric),
            out_dir=out_dir,
        )

        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch

        if args.debug:
            break

    # Predict with best checkpoint
    logging.info("Reloading best checkpoint")
    ckpt_path = out_dir / "checkpoints" / "ckpt-best.pt"
    load_checkpoint_Blob(ckpt_path, trunk, encoder, device, load_trunk=args.fine_tune)

    for split in ["testval", "test"]:
        logging.info("Generating predictions for %s", split)
        predict(
            args,
            extractor=extractor,
            encoder=encoder,
            loader=loaders[split],
            split=split,
            spaces=spaces,
            device=device,
            out_dir=out_dir,
        )

    if args.wandb:
        wandb.log(
            {"score_last": metric, "score_best": best_metric},
            step=args.epochs * epoch_steps,
        )

    logging.info("Done! Run time: %.0fs", time.monotonic() - start_time)
    logging.info("*** Best metric: %.3f (epoch %d)", best_metric, best_epoch)


@torch.no_grad()
def get_feature_shapes(
        extractor: FeatureExtractor, img_size: int, device: torch.device
):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    features, features_dict = extractor(x)
    feature_shapes = [tuple(feat.shape[1:]) for feat in features_dict.values()]

    # 遍历列表中的每个元组
    for i, size_tuple in enumerate(feature_shapes):
        # 只处理三维元组
        if len(size_tuple) == 3:
            # 计算新的第一维大小
            new_first_dim = size_tuple[0]  # 195, 384, 768, 1536

            # 计算新的第二维大小
            new_second_dim = size_tuple[1] * size_tuple[2]

            # 将元组中的值替换为新的尺寸
            feature_shapes[i] = (new_first_dim, new_second_dim)

    return feature_shapes


def set_requires_grad(
        named_params: List[Tuple[str, torch.nn.Parameter]],
        patterns: List[str],
        requires_grad: bool = False,
):
    updated = []
    for name, p in named_params:
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                p.requires_grad_(requires_grad)
                updated.append(name)
    return updated


'''
split  subject subject_id sample_id nsd_id image activity
train  subj01  0          0         13     train-0001_nsd-00013.png
train  subj01  0          1         27     train-0002_nsd-00027.png
'''


def load_dataset(args: Args, split: str) -> Dataset:
    keep_in_memory = args.cuda and torch.cuda.is_available() and not args.debug
    ds = load_from_disk((Path(args.data_dir) / "processed" / "size-256" / split),
                        keep_in_memory=keep_in_memory)
    if split == "test":
        ds = ds.select_columns(["subject_id", "nsd_id", "image"])
    else:
        ds = ds.select_columns(["subject_id", "nsd_id", "image", "activity"])

    ds.set_format("torch")
    return ds


def make_collate(transform: Callable):
    def collate_fn(batch: List[Dict[str, torch.Tensor]]):
        collated = defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                collated[k].append(v)

        # HACK: hf makes it hard to apply transforms on top of default torch formatting
        # with set_transform. So apply transform during collate.
        collated["image"] = [transform(img) for img in collated["image"]]

        collated = {k: torch.stack(v) for k, v in collated.items()}

        return collated

    return collate_fn


def create_transform(args: Args, data_config: DataConfig, train: bool = True):
    to_tensor = [ToFloatTensor()]
    resize = [
        T.Resize(
            data_config.img_size, interpolation=data_config.interp_mode, antialias=True
        ),
        # images should be square but just to be safe
        T.CenterCrop(data_config.img_size),
    ]
    normalize = [T.Normalize(mean=data_config.mean, std=data_config.std)]

    if not train:
        transform = T.Compose(to_tensor + resize + normalize)
        return transform

    augment = []
    if args.crop_scale < 1.0:
        augment.append(
            T.RandomResizedCrop(
                size=data_config.img_size,
                scale=(args.crop_scale, 1.0),
                ratio=(1.0, 1.0),
                interpolation=data_config.interp_mode,
                antialias=True,
            ),
        )
    else:
        augment.extend(resize)
    if args.jitter_prob > 0:
        augment.append(
            # Same as default in timm, no hue
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4)], p=args.jitter_prob)
        )
    if args.gray_prob > 0:
        augment.append(T.RandomGrayscale(p=args.gray_prob))
    if args.blur_prob > 0:
        augment.append(
            T.RandomApply(
                [T.GaussianBlur(kernel_size=13, sigma=(0.1, 2.0))], p=args.blur_prob
            )
        )
    if args.hflip_prob > 0:
        # This shouldn't work, screws up retinotopy
        augment.append(T.RandomHorizontalFlip(p=args.hflip_prob))

    # Erase happens after normalization due to random noise fill
    if args.re_prob > 0:
        erase = [RandomErasing(args.re_prob, mode="pixel", device="cpu")]
    else:
        erase = []
    transform = T.Compose(to_tensor + augment + normalize + erase)
    return transform


class ToFloatTensor(torch.nn.Module):
    def forward(self, img: torch.Tensor):
        img = img.to(torch.float32) / 255.0
        # HWC -> CHW
        img = torch.permute(img, (2, 0, 1)).contiguous()
        return img


def train(
        args: Args,
        epoch: int,
        extractor: FeatureExtractor,
        encoder: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Criterion,
        visualizer: Visualizer,
        device: torch.device,
        autocast: Callable,
        scaler: Optional[GradScaler],
        out_dir: Path,
):
    with_cuda = device.type == "cuda"
    if with_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Enable training mode
    encoder.train()
    # Stochastic feature evaluation
    # extractor.model.train()

    params = [p for group in optimizer.param_groups for p in group["params"]]

    def clip_grad():
        if args.clip_grad is not None and args.clip_grad > 0:
            if scaler is not None:
                # unscale the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(params, args.clip_grad).item()
        else:
            total_norm = float("nan")
        return total_norm

    loss_m = AverageMeter()
    data_time_m = AverageMeter()
    step_time_m = AverageMeter()

    examples = {}

    epoch_batches = len(train_loader)
    accum_steps = args.grad_accum_steps
    epoch_steps = math.ceil(epoch_batches / accum_steps)
    first_step = epoch * epoch_steps
    last_accum_steps = epoch_batches % accum_steps
    last_batch_idx_to_accum = epoch_batches - last_accum_steps

    # Initialize LR
    lr = update_lr(args, optimizer, first_step, epoch_steps)
    optimizer.zero_grad()

    end = time.monotonic()
    for batch_idx, sample in enumerate(train_loader):
        step = first_step + batch_idx // accum_steps
        is_last_batch = batch_idx + 1 == epoch_batches
        need_update = is_last_batch or (batch_idx + 1) % accum_steps == 0
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        # Unpack and map data to cuda
        image = sample["image"].to(device)
        target = sample["activity"].to(device)
        subidx = sample["subject_id"].to(device)
        data_time = time.monotonic() - end

        # Predict and compute loss
        _feature = []
        with autocast():
            _, features = extractor(image)

            # 遍历列表中的每个元素
            for i, value in enumerate(list(features.values())):
                # 判断张量是否是三维的
                if value.dim() == 4:
                    # 合并第二维和第三维，变为二维张量
                    reshaped_tensor = value.view(value.size(0), value.size(1), -1)
                    _feature.append(reshaped_tensor)

            pred = encoder(features=_feature, indices=subidx)  # features.values()是[128， 257，768]
            loss = criterion(pred, target)
        loss_val = loss.item()
        if accum_steps > 1:
            loss = loss / accum_steps

        if math.isnan(loss_val) or math.isinf(loss_val):
            raise RuntimeError("NaN/Inf loss encountered on step %d; exiting", step)

        # Update learning rate according to schedule
        if need_update:
            lr = update_lr(args, optimizer, step, epoch_steps)

        # Optimization step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if need_update:
                total_norm = clip_grad()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if need_update:
                total_norm = clip_grad()
                optimizer.step()

        # print(f'show grad info')
        # for name, param in encoder.named_parameters():
        #     if param.grad is not None:
        #         print(f'Layer: {name}, Gradient norm: {param.grad.norm()}')
        # print(f'END show grad info')

        if args.wandb and step % LOG_GRAD_INTERVAL == 0 and need_update:
            log_gradients(encoder)
            if args.fine_tune:
                log_gradients(extractor.model, prefix="trunk")

        # Save examples for visualization
        if args.figures and batch_idx == 0:
            pred = pred.detach().float()
            examples["image"] = image[:NUM_EXAMPLES].cpu().numpy()
            examples["target"] = target[:NUM_EXAMPLES].cpu().numpy()
            examples["pred"] = pred[:NUM_EXAMPLES].cpu().numpy()
            examples["subject_id"] = subidx[:NUM_EXAMPLES].cpu().tolist()
            examples["nsd_id"] = sample["nsd_id"][:NUM_EXAMPLES].tolist()

        # End of iteration timing
        if with_cuda:
            torch.cuda.synchronize()
        step_time = time.monotonic() - end

        loss_m.update(loss_val, len(image))
        data_time_m.update(data_time, len(image))
        step_time_m.update(step_time, len(image))

        if (step % LOG_INTERVAL == 0 and need_update) or is_last_batch or args.debug:
            tput = args.batch_size / step_time_m.avg
            alloc_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if with_cuda else 0.0
            res_mem_gb = torch.cuda.max_memory_reserved() / 1e9 if with_cuda else 0.0

            logging.info(
                f"Train: {epoch:>3d} [{batch_idx:>3d}/{epoch_batches}][{step:>6d}]"
                f"  Loss: {loss_m.val:#.3g} ({loss_m.avg:#.3g})"
                f"  LR: {lr:.3e}"
                f"  Grad: {total_norm:.3e}"
                f"  Time: {data_time_m.avg:.3f},{step_time_m.avg:.3f} {tput:.0f}/s"
                f"  Mem: {alloc_mem_gb:.2f},{res_mem_gb:.2f} GB"
            )

            record = {
                "step": step,
                "epoch": epoch,
                "loss": loss_m.val,
                "lr": lr,
                "grad": total_norm,
                "data_time": data_time_m.avg,
                "step_time": step_time_m.avg,
                "tput": tput,
            }

            with (out_dir / "train_log.json").open("a") as f:
                print(json.dumps(record), file=f)

            if args.wandb:
                wandb.log({"train": record}, step=step)

        # Restart timer for next iteration
        end = time.monotonic()

        if args.debug:
            break

    if args.figures:
        fig_dir = out_dir / "figures" / f"epoch-{epoch:02d}"  # 图像输出位置
        fig_dir.mkdir(parents=True, exist_ok=True)

        images = [topil(img) for img in examples["image"]]
        sample_scores = r2_score(
            examples["target"].T, examples["pred"].T, multioutput="raw_values"
        )
        titles = [
            f"{SUBS[subid]} nsd-{nsdid:05d} ($R^2$: {score:.3f})"
            for subid, nsdid, score in zip(
                examples["subject_id"], examples["nsd_id"], sample_scores
            )
        ]
        out_path = fig_dir / "train_pred_triplet_examples.png"
        plot_pred_triplets(
            visualizer=visualizer,
            images=images,
            targets=examples["target"],
            preds=examples["pred"],
            titles=titles,
            nrow=4,
            out_path=out_path,
        )

        if args.wandb:
            wandb_figures = {
                "train_pred_triplets": wandb.Image(str(out_path)),
                "train_pred_hist": wandb.Histogram(
                    examples["pred"].flatten(), num_bins=100
                ),
            }
            wandb.log({"figs": wandb_figures}, step=step)

        plt.close("all")


def log_gradients(model: torch.nn.Module, prefix: Optional[str] = None):
    prefix = prefix + "." if prefix else ""
    wandb_grads = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            data = p.grad.detach().flatten().cpu().float().numpy()
            wandb_grads[f"{prefix}{name}"] = wandb.Histogram(data)
    wandb.log({"gradients": wandb_grads}, commit=False)


@torch.no_grad()
def validate(
        args: Args,
        epoch: int,
        step: int,
        extractor: FeatureExtractor,
        encoder: torch.nn.Module,
        val_loader: DataLoader,
        criterion: Criterion,
        spaces: Dict[str, AlgonautsSpace],
        visualizer: Visualizer,
        device: torch.device,
        out_dir: Path,
) -> float:
    with_cuda = device.type == "cuda"
    if with_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Enable eval mode
    encoder.eval()
    extractor.model.eval()

    loss_m = AverageMeter()
    data_time_m = AverageMeter()
    step_time_m = AverageMeter()

    targets = defaultdict(list)
    preds = defaultdict(list)
    examples = {}
    nsdinds = {}

    epoch_batches = len(val_loader)
    end = time.monotonic()
    for batch_idx, sample in enumerate(val_loader):
        # Unpack and map data to cuda
        image = sample["image"].to(device)
        target = sample["activity"].to(device)
        subidx = sample["subject_id"].to(device)
        nsdidx = sample["nsd_id"]
        data_time = time.monotonic() - end

        # Predict and compute loss
        _feature = []
        _, features = extractor(image)

        # 遍历列表中的每个元素
        for i, value in enumerate(list(features.values())):
            # 判断张量是否是三维的
            if value.dim() == 4:
                # 合并第二维和第三维，变为二维张量
                reshaped_tensor = value.view(value.size(0), value.size(1), -1)
                _feature.append(reshaped_tensor)

        # Predict and compute loss
        # _, features = extractor(image)
        pred, group_pred = encoder.predict(
            features=_feature, indices=subidx
        )
        loss = criterion(pred, target)
        loss_val = loss.item()

        # Move data back to cpu for storing
        image = image.cpu()
        target = target.cpu()
        subidx = subidx.cpu()
        pred = pred.cpu().float()
        group_pred = group_pred.cpu().float()

        # Save predictions for eval
        for ii, sub in enumerate(SUBS):
            mask = subidx == ii
            count = mask.sum()
            if count > 0:
                targets[sub].append(target[mask])
                preds[sub].append(pred[mask])
                # Bc the data is unshuffled, there should be at least one batch with
                # sufficient examples per sub.
                if args.figures and sub not in examples and count >= NUM_EXAMPLES:
                    examples[sub] = image[mask][:NUM_EXAMPLES]
                    nsdinds[sub] = nsdidx[mask][:NUM_EXAMPLES].tolist()

        # Save group predictions as well
        # HACK: the name r2_group is already reserved for the total r2, so using the
        # name "base" instead.
        targets["base"].append(target)
        preds["base"].append(group_pred)
        if batch_idx == 0:
            examples["base"] = image[:NUM_EXAMPLES]
            nsdinds["base"] = nsdidx[:NUM_EXAMPLES].tolist()

        # End of iteration timing
        if with_cuda:
            torch.cuda.synchronize()
        step_time = time.monotonic() - end

        loss_m.update(loss_val, len(image))
        data_time_m.update(data_time, len(image))
        step_time_m.update(step_time, len(image))

        if (
                batch_idx % LOG_INTERVAL == 0
                or batch_idx + 1 == epoch_batches
                or args.debug
        ):
            tput = args.batch_size / step_time_m.avg
            alloc_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if with_cuda else 0.0
            res_mem_gb = torch.cuda.max_memory_reserved() / 1e9 if with_cuda else 0.0

            logging.info(
                f"Val: {epoch:>3d} [{batch_idx:>3d}/{epoch_batches}]"
                f"  Loss: {loss_m.val:#.3g} ({loss_m.avg:#.3g})"
                f"  Time: {data_time_m.avg:.3f},{step_time_m.avg:.3f} {tput:.0f}/s"
                f"  Mem: {alloc_mem_gb:.2f},{res_mem_gb:.2f} GB"
            )

        if args.debug:
            break

        # Reset timer
        end = time.monotonic()

    targets = {k: torch.cat(v).numpy() for k, v in targets.items()}
    preds = {k: torch.cat(v).numpy() for k, v in preds.items()}
    examples = {k: v.numpy() for k, v in examples.items()}

    target_scores = {}
    for sub in ["base"] + list(SUBS):
        if sub in targets:
            target_scores[sub] = raw_multi_r2(preds[sub], targets[sub])

    # 保存target_scores数据 blob和我们的

    stacked_target_scores = np.stack(
        [target_scores[sub] for sub in SUBS if sub in target_scores]
    )
    group_target_scores = np.nanmedian(stacked_target_scores, axis=0)

    r2_group = np.nanmedian(stacked_target_scores)
    sub_r2s = {sub: np.nanmedian(scores) for sub, scores in target_scores.items()}

    record = {
        "step": step,
        "epoch": epoch,
        "loss": loss_m.avg,
        "r2_group": r2_group,
        **{f"r2_{sub}": score for sub, score in sub_r2s.items()},
        "data_time": data_time_m.avg,
        "step_time": step_time_m.avg,
        "tput": tput,
    }

    with (out_dir / "val_log.json").open("a") as f:
        print(json.dumps(record), file=f)

    if args.wandb:
        wandb.log({"val": record}, step=step)

    if args.figures:
        wandb_figures = {}

        fig_dir = out_dir / "figures" / f"epoch-{epoch:02d}"
        fig_dir.mkdir(parents=True, exist_ok=True)

        map_titles = [f"group ($R^2$: {r2_group:.3f})"]
        map_titles = map_titles + [
            f"{sub} ($R^2$: {score:.3f}" for sub, score in sub_r2s.items()
        ]
        map_out_path = fig_dir / "target_scores.png"
        plot_maps(
            visualizer=visualizer,
            maps=[group_target_scores] + list(target_scores.values()),
            titles=map_titles,
            nrow=2,
            cmap="hot",
            vmin=0.0,
            vmax=0.6,
            out_path=map_out_path,
        )

        # 到此图像输出结束

        wandb_figures["target_scores"] = wandb.Image(str(map_out_path))

        for sub in examples:
            sub_images = [topil(img) for img in examples[sub]]
            sub_nsdids = nsdinds[sub]
            sub_targets = targets[sub][:NUM_EXAMPLES]
            sub_preds = preds[sub][:NUM_EXAMPLES]
            sub_sample_scores = r2_score(
                sub_targets.T, sub_preds.T, multioutput="raw_values"
            )
            titles = [
                f"{sub} nsd-{nsdid:05d} ($R^2$: {score:.3f})"
                for nsdid, score in zip(sub_nsdids, sub_sample_scores)
            ]
            sub_example_out_path = fig_dir / f"pred_triplet_examples_{sub}.png"

            plot_pred_triplets(
                visualizer=visualizer,
                images=sub_images,
                targets=sub_targets,
                preds=sub_preds,
                titles=titles,
                nrow=4,
                out_path=sub_example_out_path,
            )
            wandb_figures[f"pred_triplets_{sub}"] = wandb.Image(
                str(sub_example_out_path)
            )

        roi_tables = []
        for sub in set(SUBS).intersection(target_scores):
            space = spaces[sub]
            scores = target_scores[sub]
            table = compute_roi_scores(space, scores)

            sub_roi_out_path = fig_dir / f"roi_scores_{sub}.png"
            plot_roi_scores(
                table,
                title=f"ROI scores {sub}",
                out_path=sub_roi_out_path,
            )
            wandb_figures[f"roi_scores_{sub}"] = wandb.Image(str(sub_roi_out_path))
            roi_tables.append(table)

        roi_table = pd.concat(roi_tables, axis=0, ignore_index=True)
        roi_table.to_csv(fig_dir / "roi_table.csv", index=False)

        group_roi_table = (
            roi_table.groupby(["group", "roi", "hemi"], sort=False)
                .agg({"score": "mean"})
                .reset_index()
        )

        group_roi_out_path = fig_dir / "roi_scores_group.png"
        plot_roi_scores(
            group_roi_table,
            title="ROI scores group",
            out_path=group_roi_out_path,
        )
        wandb_figures["roi_scores_group"] = wandb.Image(str(group_roi_out_path))

        if args.wandb:
            wandb.log({"figs": wandb_figures}, step=step)

        plt.close("all")

    return r2_group


@torch.no_grad()
def predict(
        args: Args,
        extractor: FeatureExtractor,
        encoder: torch.nn.Module,
        loader: DataLoader,
        split: str,
        spaces: Dict[str, AlgonautsSpace],
        device: torch.device,
        out_dir: Path,
) -> float:
    encoder.eval()
    extractor.model.eval()

    preds = defaultdict(list)

    for sample in loader:
        image = sample["image"].to(device)
        subidx = sample["subject_id"].to(device)

        # Predict and compute loss
        _feature = []
        _, features = extractor(image)

        # 遍历列表中的每个元素
        for i, value in enumerate(list(features.values())):
            # 判断张量是否是三维的
            if value.dim() == 4:
                # 合并第二维和第三维，变为二维张量
                reshaped_tensor = value.view(value.size(0), value.size(1), -1)
                _feature.append(reshaped_tensor)

        pred = encoder(features=_feature, indices=subidx)

        subidx = subidx.cpu().numpy()
        pred = pred.cpu().float().numpy()

        for ii, sub in enumerate(SUBS):
            mask = subidx == ii
            count = mask.sum()
            if count > 0:
                preds[sub].append(pred[mask])

        if args.debug:
            break

    preds = {k: np.concatenate(v) for k, v in preds.items()}

    for sub in set(SUBS).intersection(preds):
        pred_dir = out_dir / "preds" / split / sub
        pred_dir.mkdir(parents=True)

        space = spaces[sub]
        pred = preds[sub]
        pred_lh, pred_rh = space.split_hemi(space.project(pred))
        np.save(pred_dir / f"lh_pred_{split}.npy", pred_lh)
        np.save(pred_dir / f"rh_pred_{split}.npy", pred_rh)


def update_lr(
        args: Args, optimizer: torch.optim.Optimizer, step: int, epoch_steps: int
):
    """
    Update optimizer lr according to a linear warmup + cosine decay schedule.

    Adapted from: https://github.com/karpathy/nanoGPT
    """
    total_steps = args.epochs * epoch_steps
    warmup_steps = int(args.warmup_fraction * total_steps)
    min_lr = args.min_lr_fraction * args.lr

    # Linear warmup
    if step < warmup_steps:
        lr = min_lr + (step / warmup_steps) * (args.lr - min_lr)
    # Cosine decay
    elif args.decay_lr:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = min_lr + coeff * (args.lr - min_lr)
    else:
        lr = args.lr

    # Update lr in place
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def raw_multi_r2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Drop all missing target dimensions
    # NOTE: This assumes missing dims are filled with zero in the group dataset
    mask = np.any(target != 0, axis=0)
    # NOTE: order switch
    score = r2_score(target, pred, multioutput="raw_values")
    score = np.where(mask, score, np.nan)
    return score


def compute_roi_scores(space: AlgonautsSpace, target_scores: np.ndarray):
    # Project from group to subject space if needed
    if target_scores.shape[-1] != len(space):
        target_scores = space.project(target_scores)

    scores = []
    for group in space.roi_groups():
        for roi in [None] + space.list_rois(group):
            for hemi in ["lh", "rh"]:
                mask = space.get_roi(group, roi=roi, hemi=hemi)
                score = np.nanmean(target_scores[mask]) if mask.any() else np.nan
                scores.append(
                    {
                        "sub": space.sub,
                        "group": group,
                        "roi": roi,
                        "hemi": hemi,
                        "score": score,
                    }
                )

    scores = pd.DataFrame.from_records(scores)
    return scores


def topil(img: np.ndarray) -> Image.Image:
    # CHW -> HWC
    img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())
    img = (255 * np.clip(img, 0, 1)).astype("uint8")
    return Image.fromarray(img)


def save_checkpoint(
        args: Args,
        epoch: int,
        trunk: torch.nn.Module,
        encoder: torch.nn.Module,
        metric: float,
        is_best: bool,
        out_dir: Path,
):
    state = {
        "epoch": epoch,
        "metric": metric,
        "trunk": trunk.state_dict() if args.fine_tune else None,
        "encoder": encoder.state_dict(),
    }
    ckpt_path = out_dir / "checkpoints" / "ckpt-last.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = out_dir / "checkpoints" / "ckpt-best.pt"
        torch.save(state, best_path)


def load_checkpoint_Blob(
        ckpt_path: Union[str, Path],
        trunk: torch.nn.Module,
        encoder: torch.nn.Module,
        device: torch.device,
        load_trunk: bool = True,
):
    state = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state["encoder"])

    if state.get("trunk") is not None and load_trunk:
        trunk.load_state_dict(state["trunk"])


if __name__ == "__main__":
    args: Args
    parser = HfArgumentParser(Args)
    if sys.argv[1].endswith(".yaml"):
        # If the first argument is a yaml file, parse it first to get default arguments.
        (args,) = parser.parse_yaml_file(yaml_file=sys.argv[1])

        # Treat any remaining args as overrides
        parsed = parser.parse_args(
            args=sys.argv[2:], namespace=Namespace(**asdict(args))
        )
        (args,) = parser.parse_dict(parsed.__dict__)
    else:
        (args,) = parser.parse_args_into_dataclasses()

    try:
        main(args)
    except Exception as exc:
        logging.error("Exited with exception", exc_info=exc)
        sys.exit(1)
