import os
import argparse

import torch
from src.infra.registry import SCRIPT_REGISTRY
from src.infra import config

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser('benchmark script')
    parser.add_argument('--folder', '-f', help="The path to the experiment folder", type=str, required=True)
    args = parser.parse_args()

    # load & process configs
    config_path = os.path.join(args.folder, 'config.yaml')
    opt = config.load_config(config_path)
    opt.path = args.folder
    opt.config = config_path
    print(f'loaded configrations from {config_path}')

    # torch setup
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # tf32
    if opt.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        print('tf32 enabled')

    # launch training script
    bench_script = SCRIPT_REGISTRY[opt.benchmark.script](opt)
    bench_script.run()