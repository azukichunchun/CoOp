import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.oxford_pets_active
import datasets.eurosat_active
import datasets.food101_active
import datasets.ucf101_active

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import datasets.isic
import datasets.chestxray

import trainers.coop
import trainers.cocoop
import trainers.docoop
import trainers.docoop2
import trainers.dococoop
import trainers.zsclip
import trainers.etran_score
import trainers.active
import trainers.oneshot_adapter
import trainers.clip_adapter
import trainers.oneshot_adapter_augimg
import trainers.oneshot_adapter_diverse

import random
import numpy as np


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.sample_seed:
        cfg.DATASET.SAMPLE_SEED = args.sample_seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.COOP.MIX_GEN = False
    
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.DOCOOP = CN()
    cfg.TRAINER.DOCOOP.N_CTX = 3  # number of context vectors
    cfg.TRAINER.DOCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.DOCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.DOCOOP.LAMBDA_OT = 4.0
    cfg.TRAINER.DOCOOP.LAMBDA_MI = 0.4
    cfg.TRAINER.DOCOOP.CSC = False  # class-specific context
    cfg.TRAINER.DOCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.DOCOOP.ADJUST_WEIGHT = False
    cfg.TRAINER.DOCOOP.LAMBDA_PROX = 0.000001
    cfg.TRAINER.DOCOOP.LAMBDA_CONPROX = 0.000001
    cfg.TRAINER.DOCOOP.MIX_GEN = False
    cfg.TRAINER.DOCOOP.LR_PROX = 0.001
    cfg.TRAINER.DOCOOP.LR_CONPROX = 0.0001
    
    cfg.TRAINER.DOCOOP2 = CN()
    cfg.TRAINER.DOCOOP2.N_CTX = 3  # number of context vectors
    cfg.TRAINER.DOCOOP2.N_DMX = 3  # number of context vectors
    cfg.TRAINER.DOCOOP2.CTX_INIT = ""  # initialization words
    cfg.TRAINER.DOCOOP2.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.DOCOOP2.LAMBDA_OT = 4.0
    cfg.TRAINER.DOCOOP2.LAMBDA_MI = 0.4
    cfg.TRAINER.DOCOOP2.CSC = False  # class-specific context
    cfg.TRAINER.DOCOOP2.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.DOCOOP2.ADJUST_WEIGHT = False
    
    cfg.TRAINER.DOCOCOOP = CN()
    cfg.TRAINER.DOCOCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.DOCOCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.DOCOCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.DOCOCOOP.LAMBDA_OT = 4.0
    cfg.TRAINER.DOCOCOOP.LAMBDA_MI = 0.4
    cfg.TRAINER.DOCOCOOP.CSC = False  # class-specific context
    cfg.TRAINER.DOCOCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.DOCOCOOP.ADJUST_WEIGHT = False
    cfg.TRAINER.DOCOCOOP.LAMBDA_PROX = 0.000001
    cfg.TRAINER.DOCOCOOP.LAMBDA_CONPROX = 0.000001
    cfg.TRAINER.DOCOCOOP.LR_PROX = 0.001
    cfg.TRAINER.DOCOCOOP.LR_CONPROX = 0.0001
    cfg.TRAINER.DOCOCOOP.PROX_EPOCH = 12
    
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.NUM_CLASS = 10
    cfg.DATASET.SAMPLE_SEED = 1

    cfg.DATALOADER.ENERGY = CN()
    cfg.DATALOADER.ENERGY.USE_ENERGY = False
    cfg.DATALOADER.ENERGY.USAGE_RANK = "max"

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()    

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

        # check random seed #
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sample = random.sample(data, 3)
        print(f"When seed is {cfg.SEED}, choised samples are {sample}")

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    trainer = build_trainer(cfg)
    if args.embedding_feature:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.embedding_feature()
        return
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    if args.eval_only and cfg.TEST.FINAL_MODEL == "best_val":
        trainer.load_model(args.model_dir)
        trainer.test()
        return
    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--sample-seed", type=int, default=-1, help="only positive value enables a fixed sample seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--embedding_feature", action="store_true", help="embedding feature using tsne")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
