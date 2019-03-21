import argparse
import inspect
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import sys
project_dir = Path(__file__).resolve().parents[2]
# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())
sys.path.append(str(project_dir / "src"))
from utils import timer


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) \
                and not inspect.isabstract(v):
            yield v()


def generate_features(features, overwrite):
    for f in features:
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub(
                "([A-Z])",
                lambda x: "_" + x.group(1).lower(), self.__class__.__name__
            ).lstrip('_')

        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.feather'
        self.test_path = Path(self.dir) / f'{self.name}_test.feather'

    def run(self):
        #with timer(self.name):
        # import pdb; pdb.set_trace()
        self.create_features()
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.train.columns = prefix + self.train.columns + suffix
        self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))

    def load(self):
        self.train = pd.read_feather(str(self.train_path))
        self.test = pd.read_feather(str(self.test_path))