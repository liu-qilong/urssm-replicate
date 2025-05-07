import torch
import argparse
from pathlib import Path

from src.tool.registry import SCRIPT_REGISTRY
from src.tool import config

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser('training script')
    parser.add_argument('--path', '-p', help="The path to the experiment folder where the configuration sheet, network weights, and other results are stored.", type=str, required=True)
    args = parser.parse_args()

    # load options
    opt = config.load_config(args.path)
    opt.path = args.path

    print(f'loaded configrations from {Path(args.path) / "config.yaml"}')
    print('-'*50)

    # torch setup
    torch.manual_seed(0)

    # launch training script
    train_script = SCRIPT_REGISTRY[opt.train.script](opt)
    train_script.load_data()
    train_script.train_prep()
    train_script.train_loop()