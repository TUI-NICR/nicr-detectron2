import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)  # noqa
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))  # noqa

import os
from datetime import datetime
from detectron2.utils.events import EventStorage

import torch

from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch
)
from detectron2.evaluation import verify_results

from nicr_detectron2.evaluation.grasp_evaluator import GraspVisualizationEvaluator, GraspNetPredictionSaver

import nicr_detectron2.modeling
from nicr_detectron2.config.default_config import add_grasp_config
from nicr_detectron2.data import set_datasets_root

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from nicr_grasping import graspnet_dataset_path

from nicr_detectron2.engine.grasp_net_trainer import GraspNetTrainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = add_grasp_config(cfg)
    cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    if args.debug:
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_debug'

    set_datasets_root(args.dataset_root)

    training_starttime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S-%f")

    if not args.eval_only:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, training_starttime)
    else:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR,
                                      'eval',
                                      training_starttime)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    cfg = setup(args)

    graspnet_path = graspnet_dataset_path()

    if args.eval_only:
        with EventStorage():
            model = GraspNetTrainer.build_model(cfg)
            model.to(torch.device(cfg.MODEL.DEVICE))
            DetectionCheckpointer(
                model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = GraspNetTrainer.test(cfg, model)
            if comm.is_main_process():
                verify_results(cfg, res)
            return res

    trainer = GraspNetTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    if cfg.MODEL.META_ARCHITECTURE == "UncertainGraspNet":
        trainer.model.load_pretrained_weights()
    trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--json_root', type=str,
                        help='path to root of dataset json files')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset_root', type=str,
                        help='path to root of dataset',
                        default=None)
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
