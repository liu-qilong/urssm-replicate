import os
import time
import shutil
import argparse
from pathlib import Path

import torch
from src.infra.registry import SCRIPT_REGISTRY
from src.infra import config

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser('training script')
    parser.add_argument('--config', '-c', help="The path to the configuration file", type=str, required=True)
    args = parser.parse_args()

    # load & process configs
    opt = config.load_config(args.config)
    print(f'loaded configrations from {args.config}')

    opt.path = os.path.join('experiment', f'{Path(args.config).stem}-{time.strftime("%Y%m%d-%H%M%S")}')
    Path(opt.path).mkdir()
    print(f'created experiment folder {opt.path}')

    shutil.copy(args.config, Path(opt.path) / Path(args.config).name)
    print(f"copied configurations to {Path(opt.path) / 'config.yaml'}")
    print('-' * 50)

    # torch setup
    torch.manual_seed(0)

    # launch training script
    train_script = SCRIPT_REGISTRY[opt.train.script](opt)
    train_script.load_data()
    train_script.train_prep()
    train_script.train_loop()