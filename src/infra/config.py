from pathlib import Path
from yacs.config import CfgNode

def load_config(path):
    opt_copy = CfgNode()
    opt_copy.set_new_allowed(True)
    opt_copy.merge_from_file(path)
    
    return opt_copy