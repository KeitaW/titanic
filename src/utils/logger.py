import logging
from lightgbm.callback import _format_eval_result
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[2]
# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

"""
REF: https://github.com/upura/ml-competition-template-titanic/blob/master/logs/logger.py
http://icebee.hatenablog.com/entry/2018/12/16/221533
"""


def create_logger(logfile):
    log_file = (project_dir / "reports/logs/{}.log".format(logfile)).resolve()

    # logger
    logger_ = getLogger(logfile)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger(logfile):
    return getLogger(logfile)

def log_best(model, metric):
    logging.debug(model.best_iteration)
    logging.debug(model.best_score['valid_0'][metric])


def lgbm_logger(logger, period=1, show_stdv=True, level=logging.DEBUG):
    """ Create a callback that prints the evaluation results which intended to be used with LightGBM
    Parameters
    ----------
    period : int, optional (default=1)
        The period to print the evaluation results.
    show_stdv : bool, optional (default=True)
        Whether to show stdv (if provided).
    Returns
    -------
    callback : function
        The callback that prints the evaluation results every ``period`` iteration(s).
    
    REF
    ------------
    https://amalog.hateblo.jp/entry/lightgbm-logging-callback
    https://upura.hatenablog.com/entry/2018/12/28/225234
    """
    def _callback(env):
        if period > 0 and env.evaluation_result_list \
                and (env.iteration + 1) % period == 0:
            result = '\t'.join([
                _format_eval_result(x, show_stdv)
                for x in env.evaluation_result_list
            ])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback