import os
import logging

from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboard import default, program
import tensorflow as tf

from utils import get_dirname_from_args, get_now


class Logger:
    def __init__(self, args):
        self.log_cmd = args.log_cmd
        log_name = get_dirname_from_args(args)
        log_name += f'_{get_now()}'
        self.log_path = args.log_path / log_name
        os.makedirs(self.log_path, exist_ok=True)
        self.tfboard = SummaryWriter(self.log_path)

        print(f"Logging at {self.log_path}")
        self.url = run_tensorboard(self.log_path)
        url = self.url.split(':')
        args.hostname = args.get('hostname', 'localhost')
        url = f"{':'.join(url[:-1])}.{args.hostname}:{url[-1]}"

        print(f"Running Tensorboard at {url}")

    def __call__(self, name, val, n_iter):
        if isinstance(val, str):
            # https://stackoverflow.com/questions/45016458/tensorflow-tf-summary-text-and-linebreaks
            val = val.replace('\n', '  \n')
            self.tfboard.add_text(name, val, n_iter)
            if self.log_cmd:
                tqdm.write(f'{n_iter}:({name},{val})')
        elif isinstance(val, int) or isinstance(val, float):
            self.tfboard.add_scalar(name, val, n_iter)
            if self.log_cmd:
                tqdm.write(f'{n_iter}:({name},{val})')
        else:
            print(f"unloggable type: {type(val)}/{val}")


def run_tensorboard(log_path):
    log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
    log = logging.getLogger('tensorflow').setLevel(logging.ERROR)

    port_num = abs(hash(log_path)) % (8800) + 1025  # above 1024, below 10000
    tb = program.TensorBoard(default.get_plugins(), get_assets_zip_provider())
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
        print('webfiles.zip static assets not found: %s', path)
        return None
  return lambda: open(path, 'rb')


def log_results(logger, name, state, step):
    for key, val in state.metrics.items():
        if isinstance(val, dict):
            for key2, v in val.items():
                logger(f"{name}/{key}/{key2}" , v, step)
        else:
            logger(f"{name}/{key}" , val, step)


def get_logger(args):
    return Logger(args)
