"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *
from lavis.layers.nbitlineardynamic import NBitLinearDynamic

# @vla: custom parse options and quantize function
from lavis.args_parser import parse_args
from lavis.quantize import *

# def parse_args():
#     parser = argparse.ArgumentParser(description="Training")

#     parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )
    
#     parser.add_argument('--img-submodule-FF-weight_bits', required = False, default = None, type = int)
#     parser.add_argument('--img-submodule-FF-activation_bits', required = False, default = None, type = int)
    
#     parser.add_argument('--text-submodule-FF-weight_bits', required = False, default = None, type = int)
#     parser.add_argument('--text-submodule-FF-activation_bits', required = False, default = None, type = int)

#     args = parser.parse_args()
#     # if 'LOCAL_RANK' not in os.environ:
#     #     os.environ['LOCAL_RANK'] = str(args.local_rank)

#     return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    
    args = parse_args()
    cfg = Config(args)
  
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # APPLY QUANTIZATION CONFIG
    quantize(model, args)
    
    print(model)
    
    # write out model size
    size = model_size(model)
    print(f'[Model Size]: {size}')
    
    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.evaluate(skip_reload=True)

if __name__ == "__main__":
    main()
