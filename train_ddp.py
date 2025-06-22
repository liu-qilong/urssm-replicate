import os
import time
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from src.infra.registry import SCRIPT_REGISTRY
from src.infra import config

def run(rank, opt):
    world_size = opt.train.world_size
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=opt.train.backend,
    )
    train_script = SCRIPT_REGISTRY[opt.train.script](opt, rank, world_size)
    train_script.run()

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

    # launch training script
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(opt.train.ddp_port)
    world_size = opt.train.world_size
    mp.spawn(
        run,
        args=(opt,),
        nprocs=world_size,
        join=True,
    )