"""Optimizes hyperparameters using Bayesian optimization."""

from copy import deepcopy
import json
from typing import Dict, Union
import os

from hyperopt import fmin, hp, tpe
import numpy as np

from HMSA.args import HyperoptArgs
from HMSA.constants import HYPEROPT_LOGGER_NAME
from HMSA.models import HMSAModel
from HMSA.nn_utils import param_count
from HMSA.train import cross_validate, run_training
from HMSA.utils import create_logger, makedirs, timeit


SPACE = {
    'hidden_size': hp.quniform('hidden_size', low=300, high=2400, q=100),
    'depth': hp.quniform('depth', low=2, high=6, q=1),
    'dropout': hp.quniform('dropout', low=0.0, high=0.4, q=0.05),
    'ffn_num_layers': hp.quniform('ffn_num_layers', low=1, high=3, q=1)
}
INT_KEYS = ['hidden_size', 'depth', 'ffn_num_layers']


@timeit(logger_name=HYPEROPT_LOGGER_NAME)
def hyperopt(args: HyperoptArgs) -> None:
    logger = create_logger(name=HYPEROPT_LOGGER_NAME, save_dir=args.log_dir, quiet=True)

    results = []

    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        hyper_args = deepcopy(args)

        if args.save_dir is not None:
            folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items())
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)

        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)

        hyper_args.ffn_hidden_size = hyper_args.hidden_size

        logger.info(hyperparams)

        mean_score, std_score = cross_validate(args=hyper_args, train_func=run_training)

        temp_model = HMSAModel(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f'num params: {num_params:,}')
        logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')

        results.append({
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams,
            'num_params': num_params
        })

        if np.isnan(mean_score):
            if hyper_args.dataset_type == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')

        return (1 if hyper_args.minimize_score else -1) * mean_score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters, rstate=np.random.RandomState(args.seed))

    results = [result for result in results if not np.isnan(result['mean_score'])]
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['mean_score'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(f'{best_result["mean_score"]} +/- {best_result["std_score"]} {args.metric}')

    makedirs(args.config_save_path, isfile=True)

    with open(args.config_save_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


def chemprop_hyperopt() -> None:
    hyperopt(args=HyperoptArgs().parse_args())
