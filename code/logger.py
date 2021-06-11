import os
import logging
import json
from collections import defaultdict

from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboard import default, program
import tensorflow as tf

from exp import ex
from path import get_dirname_from_args
from utils import get_now, recursive_insert_dict


class Logger:
    @ex.capture
    def __init__(self, logger_type, log_multi, log_path, hostname, _config):
        self.log_multi = log_multi
        self.log_path = self.get_log_path(log_path)
        self.hostname = hostname
        self.log_data = defaultdict(dict)
        self.args = _config
        self.save_args()

        self.logger_dests = []
        if len(ex.observers) > 0:
            print("using sacred logger")
            self.logger_dests.append('sacred')
        for logger in logger_type:
            getattr(self, f'init_{logger}')()

    def get_log_path(self, log_path):
        log_name = get_dirname_from_args()
        log_name += f'_{get_now()}'
        self.log_path = log_path / log_name
        os.makedirs(self.log_path, exist_ok=True)
        return self.log_path

    def save_args(self):
        with open(self.log_path / 'args.json', 'w') as f:
            args = {str(k): str(v) for k, v in self.args.items()}
            json.dump(args, f, indent=4)

    def save_log_data(self):
        with open(self.log_path / 'epoch_stats.json', 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def init_tfboard(self):
        self._init_tfboard(self.log_path, self.hostname)
        self.logger_dests.append('tfboard')

    def init_cmd(self):
        self.logger_dests.append('cmd')

    def init_file(self):
        self._init_file()
        self.logger_dests.append('file')

    def _init_file(self):
        self.log_file = open(self.log_path / 'log.txt', 'w')

    def _init_tfboard(self, log_path, hostname):
        self.tfboard = SummaryWriter(str(self.log_path))

        print(f"Logging at {self.log_path}")
        '''
        self.url = run_tensorboard(self.log_path if not self.log_multi else log_path)
        url = self.url.split(':')
        url = f"{':'.join(url[:-1])}.{hostname}:{url[-1]}"

        print(f"Running Tensorboard at {url}")
        '''

    def log_text(self, name, val, n_iter):
        for logger_name in self.logger_dests:
            getattr(self, f"log_text_{logger_name}")(name, val, n_iter)

    def log_text_tfboard(self, name, val, n_iter):
        # https://stackoverflow.com/questions/45016458/tensorflow-tf-summary-text-and-linebreaks
        val = val.replace('\n', '  \n')
        self.tfboard.add_text(name, val, n_iter)

    def log_text_cmd(self, name, val, n_iter):
        tqdm.write(f'{n_iter}:({name},{val})')

    def log_text_file(self, name, val, n_iter):
        self.log_file.write(f'{n_iter}:({name},{val})\n')

    @ex.capture
    def log_text_sacred(self, name, val, n_iter, _run):
        name = ['text', *name.split('/'), str(n_iter)]
        val = val.split('\n')
        val = [v.strip() for v in val]
        val = [v for v in val if v]
        _run.info = recursive_insert_dict(_run.info, name, val)
        # _run.info[f"{name}:{n_iter}"] = val

    def log_epoch(self, name, val, n_iter):
        self.log_data[name][n_iter] = val
        self.save_log_data()

    def log_scalar(self, name, val, n_iter):
        if len(list(part for part in name.split('/') if part == 'epoch')) > 0:
            self.log_epoch(name, val, n_iter)
        for logger_name in self.logger_dests:
            getattr(self, f"log_scalar_{logger_name}")(name, val, n_iter)

    def log_scalar_tfboard(self, name, val, n_iter):
        self.tfboard.add_scalar(name, val, n_iter)

    def log_scalar_cmd(self, name, val, n_iter):
        tqdm.write(f'{n_iter}:({name},{val})')

    def log_scalar_file(self, name, val, n_iter):
        self.log_file.write(f'{n_iter}:({name},{val})\n')

    @ex.capture
    def log_scalar_sacred(self, name, val, n_iter, _run):
        _run.log_scalar(name.replace('/', '.'), val, n_iter)

    def __call__(self, name, val, n_iter):
        if isinstance(val, str):
            self.log_text(name, val, n_iter)
        elif isinstance(val, int) or isinstance(val, float):
            self.log_scalar(name, val, n_iter)
        else:
            print(f"unloggable type: {type(val)}/{val}")

    def __del__(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()


def run_tensorboard(log_path):
    log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
    log = logging.getLogger('tensorflow').setLevel(logging.ERROR)

    port_num = abs(hash(log_path)) % (8800) + 1025  # above 1024, below 10000
    tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
    tb.configure(argv=[None, '--logdir', str(log_path), '--port', str(port_num),
                       '--samples_per_plugin', 'text=100'])
    # samples_per_plugin keeps tensorboard from skipping texts for random steps
    url = tb.launch()
    return url


# forward compatibility for version > 1.12
def get_assets_zip_provider():
  """Opens stock TensorBoard web assets collection.
  Returns:
    Returns function that returns a newly opened file handle to zip file
    containing static assets for stock TensorBoard, or None if webfiles.zip
    could not be found. The value the callback returns must be closed. The
    paths inside the zip file are considered absolute paths on the web server.
  """
  path = os.path.join(tf.resource_loader.get_data_files_path(), 'webfiles.zip')
  if not os.path.exists(path):
        # print('webfiles.zip static assets not found: %s', path)
        return None
  return lambda: open(path, 'rb')


def log_results(logger, name, state, step):
    for key, val in state.metrics.items():
        if isinstance(val, dict):
            for key2, v in val.items():
                logger(f"{name}/{key}/{key2}" , v, step)
        else:
            logger(f"{name}/{key}" , val, step)


@ex.capture
def get_logger(logger_type, log_multi, log_path, hostname, log_file=False):
    if log_file:
        logger_type = ['file']
    return Logger(logger_type, log_multi, log_path, hostname)
