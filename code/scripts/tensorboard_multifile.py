import os
import logging
from pathlib import Path

from tensorboard import default, program
import tensorflow as tf

from exp import ex


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


@ex.capture
def main(logdir):
    path = Path(logdir).resolve()
    paths = path.parent.glob(path.name)
    k = 'keyword_name_keywords'
    string = []
    for p in paths:
        name = p.name
        name = name[name.find(k) + len(k):]
        name = name[:name.find('_')]
        string.append(f"{name}:{p}")
    string = ','.join(string)

    run_tensorboard(string)


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


if __name__ == '__main__':
    main()
