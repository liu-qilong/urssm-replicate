import os.path as osp
import importlib
from src.util.misc import scandir

# automatically import all sub modules under the current directory
folder = osp.dirname(osp.abspath(__file__))
folder_name = osp.basename(folder)

modules = []
for v in scandir(folder, recursive=True, full_path=False, suffix='.py'):
    if v.endswith('__init__.py'):
        # when __init__.py is found, import the parent directory, except for the current one
        v = osp.dirname(v)
        if v == '':
            continue

    module_name = folder_name + '.' + v.replace('/', '.').replace('.py', '')

    try:
        module = importlib.import_module(module_name)
        modules.append(module)
        # print(f"successfully imported {module_name}")
    except ModuleNotFoundError as e:
        print(f"failed to import {module_name}: {e}")