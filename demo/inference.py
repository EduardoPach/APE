# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import requests
from collections import abc

import cv2
from PIL import Image
import numpy as np
import tqdm

from detectron2.config import LazyConfig, get_cfg

# from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo
from ape.engine.defaults import DefaultPredictor

# constants
WINDOW_NAME = "APE"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    if "output_dir" in cfg.model:
        cfg.model.output_dir = cfg.train.output_dir
    if "model_vision" in cfg.model and "output_dir" in cfg.model.model_vision:
        cfg.model.model_vision.output_dir = cfg.train.output_dir
    if "train" in cfg.dataloader:
        if isinstance(cfg.dataloader.train, abc.MutableSequence):
            for i in range(len(cfg.dataloader.train)):
                if "output_dir" in cfg.dataloader.train[i].mapper:
                    cfg.dataloader.train[i].mapper.output_dir = cfg.train.output_dir
        else:
            if "output_dir" in cfg.dataloader.train.mapper:
                cfg.dataloader.train.mapper.output_dir = cfg.train.output_dir

    if "model_vision" in cfg.model:
        cfg.model.model_vision.test_score_thresh = args.confidence_threshold
    else:
        cfg.model.test_score_thresh = args.confidence_threshold

    # default_setup(cfg, args)

    setup_logger(name="ape")
    setup_logger(name="timm")

    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl.py",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[
            "train.init_checkpoint=./model_final.pth",
            "train.device=cpu",
            "model.model_language.cache_dir=''",
            "model.model_vision.select_box_nums_for_evaluation=500",
            "model.model_vision.text_feature_bank_reset=True"
        ],
        nargs=argparse.REMAINDER,
    )

    return parser


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im

if __name__ == "__main__":
    args = get_parser().parse_args()

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args=args)
    # model = DefaultPredictor(cfg)

    img = prepare_img()
    img_array = np.array(img)
    text_prompt = "a cat sitting on a couch with remote controls"
    with_box = True
    with_mask = True
    with_sseg = True

    # predictions = model(img_array, text_prompt)

    predictions, visualized_output, visualized_outputs, metadata = demo.run_on_image(
        img_array,
        text_prompt=text_prompt,
        with_box=with_box,
        with_mask=with_mask,
        with_sseg=with_sseg,
    )