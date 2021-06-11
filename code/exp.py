import inspect
from pathlib import Path
import json

import sacred
from sacred.utils import optional_kwargs_decorator
from sacred.config.signature import Signature
from sacred.config.captured_function import create_captured_function
from sacred.observers import MongoObserver, SlackObserver

#torch.multiprocessing.set_sharing_strategy('file_system')  # https://github.com/pytorch/pytorch/issues/973


class LazyCapture:
    def __init__(self, function=None, prefix=None):
        self.fn = function
        self.prefix = prefix
        self.__name__ = self.fn.__name__
        self.signature = Signature(self.fn)

    def update_obj(self, wrap):
        self.config = wrap.config
        self.rnd = wrap.rnd
        self.run = wrap.run
        self.logger = wrap.logger
        self.signature = wrap.signature

    def update_func(self, cfn):
        cfn.config = self.config
        cfn.rnd = self.rnd
        cfn.run = self.run
        cfn.logger = self.logger
        cfn.signature = self.signature
        return cfn

    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'lazy_fn'):
            self.lazy_fn = create_captured_function(self.fn, prefix=self.prefix)
            self.lazy_fn = self.update_func(self.lazy_fn)
        return self.lazy_fn(*args, **kwargs)


def wrap_capture(function=None, prefix=None):
    lazy_capture = LazyCapture(function, prefix=prefix)

    def captured_function(*args, **kwargs):
        lazy_capture.update_obj(captured_function)
        return lazy_capture(*args, **kwargs)

    captured_function.signature = lazy_capture.signature
    captured_function.prefix = lazy_capture.prefix

    return captured_function


class Experiment(sacred.Experiment):
    @optional_kwargs_decorator
    def capture(self, function=None, prefix=None):
        if function in self.captured_functions:
            return function
        captured_function = wrap_capture(function, prefix=prefix)
        self.captured_functions.append(captured_function)
        return captured_function


exp_name = 'lsmdc'
ex = Experiment(exp_name)


def load_config(path):
    path = Path(path).resolve()
    if path.is_file():
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return None


def add_observers():
    obs_config = load_config('../observers/config.json')
    if obs_config is not None and obs_config['observe']:
        if 'mongo' in obs_config and obs_config['mongo']:
            port = obs_config['port']
            hostname = obs_config.get('hostname', 'localhost')
            username = obs_config['username']
            password = obs_config['password']
            print(f"loading mongo observer at {hostname}:{port}")
            ex.observers.append(
                MongoObserver.create(
                    url=f'mongodb://{username}:{password}@{hostname}:{port}/{exp_name}',
                    db_name=exp_name))
        if 'slack' in obs_config and obs_config['slack']:
            print("loading slack observer")
            slack_path = Path('../observers/slack.json').resolve()
            if slack_path.is_file():
                slack_obs = SlackObserver.from_config(str(slack_path))
                ex.observers.append(slack_obs)


add_observers()
