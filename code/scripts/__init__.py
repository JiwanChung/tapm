import os
from pathlib import Path


script_dict = {}


def add_scripts():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__(f"{parent}.{name}")
            module = eval(name)
            if hasattr(module, 'main'):
                script_dict[str(name)] = module.main


def run_script(name):
    print(f'running script {name}')
    if not script_dict:
        add_scripts()
    if name in script_dict:
        script_dict[name]()
    else:
        print(f'no script named {name}')
