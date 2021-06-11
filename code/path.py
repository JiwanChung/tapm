import re
from pathlib import Path

from exp import ex


def add_keyword_paths(data_path, keyword_dir):
    keyword_dir = resolve_keyword_dir(keyword_dir, list(data_path.values())[0])
    for key, path in data_path.items():
        data_path[key] = get_keyword_path(data_path, key, keyword_dir)
    return data_path


def resolve_keyword_dir(keyword_dir, path):
    candidates = list(path.parent.glob(f'{keyword_dir}*'))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        x = candidates[0]
        print(f"choosing keyword path {x}")
        return x
    else:
        assert len(candidates) > 0, \
            f"no candidate for keyword dir: {keyword_dir}"


def get_keyword_path(data_path, key, dir_name=None):
    if dir_name is None:
        dir_name = f'keywords_{get_dirname_from_args()}'
    path = data_path[key].parent
    if path.name.startswith('keyword'):
        path = path.parent
    path =  path / dir_name
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{data_path[key].name}.json"
    return path


@ex.capture
def get_dirname_from_args(_config):
    dirname = ''
    for key in sorted(_config['log_keys']):
        if key != 'log_keys':
            dirname += '_'
            dirname += key
            dirname += '_'
            val = _config[key]
            if isinstance(val, float) and (not key == 'learning_rate'):
                val = '{:.2f}'.format(val)
            else:
                val = re.sub('[,\W-]+', '_', str(val))
            dirname += val

    dirname = dirname[1:]
    dirname = dirname[:100]  # avoid file name too long error

    return dirname
