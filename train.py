import os
import time
import argparse

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
    opt.path = os.path.join('experiment', f'{opt.exp_name}-{time.strftime("%Y%m%d-%H%M%S")}')
    opt.config = args.config
    print(f'loaded configrations from {args.config}')

    # torch setup
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # tf32
    if opt.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        print('tf32 enabled')

    # launch training script
    train_script = SCRIPT_REGISTRY[opt.train.script](opt)
    train_script.run()