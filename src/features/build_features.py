# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sys
from inspect import isclass
# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[2]
# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())
sys.path.append(str(project_dir / "src"))
from utils.path import get_suffix
from utils.logger import create_logger, get_logger
from features.base import Feature, generate_features, get_features
import pandas as pd
import numpy as np
import re as re

Feature.dir = 'data/processed'


class Pclass(Feature):
    def create_features(self):
        self.train['Pclass'] = train['Pclass']
        self.test['Pclass'] = test['Pclass']


class Sex(Feature):
    def create_features(self):
        self.train['Sex'] = train['Sex'].replace(['male', 'female'], [0, 1])
        self.test['Sex'] = test['Sex'].replace(['male', 'female'], [0, 1])


class FamilySize(Feature):
    def create_features(self):
        self.train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
        self.test['FamilySize'] = test['Parch'] + test['SibSp'] + 1


class Embarked(Feature):
    def create_features(self):
        self.train['Embarked'] = train['Embarked'] \
            .fillna(('S')) \
            .map({'S': 0, 'C': 1, 'Q': 2}) \
            .astype(int)
        self.test['Embarked'] = test['Embarked'] \
            .fillna(('S')) \
            .map({'S': 0, 'C': 1, 'Q': 2}) \
            .astype(int)


class Fare(Feature):
    def create_features(self):
        data = train.append(test)
        fare_mean = data['Fare'].mean()
        self.train['Fare'] = pd.qcut(
            train['Fare'].fillna(fare_mean),
            4,
            labels=False,
            duplicates='drop'
        )
        self.test['Fare'] = pd.qcut(
            test['Fare'].fillna(fare_mean),
            4,
            labels=False,
            duplicates='drop'
        )


class Age(Feature):
    def create_features(self):
        data = train.append(test)
        age_mean = data['Age'].mean()
        age_std = data['Age'].std()
        self.train['Age'] = pd.qcut(
            train['Age'].fillna(
                np.random.randint(age_mean - age_std, age_mean + age_std)
            ),
            5,
            labels=False
        )
        self.test['Age'] = pd.qcut(
            test['Age'].fillna(
                np.random.randint(age_mean - age_std, age_mean + age_std)
            ),
            5,
            labels=False
        )


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


class Title(Feature):
    def create_features(self):
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        self.train['Title'] = train['Name'] \
            .apply(get_title) \
            .replace([
                'Lady',
                'Countess',
                'Capt',
                'Col',
                'Don',
                'Dr',
                'Major',
                'Rev',
                'Sir',
                'Jonkheer',
                'Dona'
            ], 'Rare') \
            .replace(['Mlle', 'Ms', 'Mme'], ['Miss', 'Miss', 'Mrs'])
        self.train['Title'] = train['Name'].map(title_mapping).fillna(0)
        self.test['Title'] = test['Name'] \
            .apply(get_title) \
            .replace([
                'Lady',
                'Countess',
                'Capt',
                'Col',
                'Don',
                'Dr',
                'Major',
                'Rev',
                'Sir',
                'Jonkheer',
                'Dona'
            ], 'Rare') \
            .replace(['Mlle', 'Ms', 'Mme'], ['Miss', 'Miss', 'Mrs'])
        self.test['Title'] = self.test['Title'].map(title_mapping).fillna(0)
        self.train['Title'] = self.train['Title']
        self.test['Title'] = self.test['Title']
train = None
test = None
@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--force', is_flag=True)
def main(train_file, test_file, force):
    global train
    global test
    train = pd.read_feather(train_file)
    test = pd.read_feather(test_file)
    features = list(get_features(globals()))
    generate_features(features, force)
    
VERSION="0.0.1"
if __name__ == '__main__':
    create_logger(VERSION)
    get_logger(VERSION).info("Start feature generation.")
    main()
