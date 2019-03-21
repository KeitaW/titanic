# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import sys
# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[2]
# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())
sys.path.append(str(project_dir / "src"))
from utils.path import get_suffix
from utils.logger import create_logger, get_logger

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def raw2feather(input_filepath, output_filepath):
    if get_suffix(input_filepath) in ['csv', 'tsv', 'gzip', 'bz2', 'zip', 'xz']:
        assert("Unsupported file format.")
    pd.read_csv(
       input_filepath 
    ).to_feather(output_filepath)
    
def main():
    raw2feather()
    
if __name__ == '__main__':
    logfile = "raw2feather"
    create_logger(logfile)
    get_logger(logfile).info("Convert file")
    main()
    get_logger(logfile).info("Saved")
    