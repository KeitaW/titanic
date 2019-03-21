import time
from functools import wraps
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sys
# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[2]
# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())
sys.path.append(str(project_dir / "src"))
from utils.logger import get_logger

"""
REF: http://icebee.hatenablog.com/entry/2018/12/16/221533
"""
def stop_watch(VERSION):

    def _stop_watch(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            start = time.time()

            result = func(*args, **kargs)

            elapsed_time = int(time.time() - start)
            minits, sec = divmod(elapsed_time, 60)
            hour, minits = divmod(minits, 60)

            get_logger(VERSION).info("[elapsed_time]\t>> {:0>2}:{:0>2}:{:0>2}".format(hour, minits, sec))
        return wrapper

    return _stop_watch