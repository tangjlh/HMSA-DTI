from collections import defaultdict
import csv
import json
from logging import Logger
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .run_training import run_training
from HMSA.args import TrainArgs
from HMSA.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from HMSA.data import get_data, get_task_names, MoleculeDataset, validate_dataset_type
from HMSA.utils import create_logger, makedirs, timeit
from HMSA.features import set_explicit_h, set_reaction
from tape import TAPETokenizer


@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args: TrainArgs,
                   train_func: Callable[[TrainArgs, MoleculeDataset, Logger], Dict[str, List[float]]]
                   ) -> Tuple[float, float]:

    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    makedirs(args.save_dir)
    args.save(os.path.join(args.save_dir, 'args.json'), with_reproducibility=False)

    set_explicit_h(args.explicit_h)
    set_reaction(args.reaction, args.reaction_mode)

    tokenizer = TAPETokenizer(vocab='unirep')
        
    debug('Loading data')
    data = get_data(
        path=args.data_path,
        args=args,
        smiles_columns=args.smiles_columns,
        logger=logger,
        skip_none_targets=True
    )
    validate_dataset_type(data, dataset_type=args.dataset_type)
    args.features_size = data.features_size()

    if args.target_weights is not None and len(args.target_weights) != args.num_tasks:
        raise ValueError('The number of provided target weights must match the number and order of the prediction tasks')

    all_scores = defaultdict(list)
    Cindexs = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        data.reset_features_and_targets()

        test_scores_path = os.path.join(args.save_dir, 'test_scores.json')
        if args.resume_experiment and os.path.exists(test_scores_path):
            print('Loading scores')
            with open(test_scores_path) as f:
                model_scores = json.load(f)
        else:
            model_scores,Cindex = train_func(args, data, fold_num, logger, tokenizer)
        Cindexs.append(Cindex)
        for metric, scores in model_scores.items():
            all_scores[metric].append(scores)
    all_scores = dict(all_scores)

    for metric, scores in all_scores.items():
        all_scores[metric] = np.array(scores)

    info(f'{args.num_folds}-fold cross validation')

    for fold_num in range(args.num_folds):
        for metric, scores in all_scores.items():
            info(f'\tSeed {init_seed + fold_num} ==> test {metric} = {np.nanmean(scores[fold_num]):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(args.task_names, scores[fold_num]):
                    info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} {metric} = {score:.6f}')

    for metric, scores in all_scores.items():
        avg_scores = np.nanmean(scores, axis=1)
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        cimean_score, cistd_score = np.nanmean(Cindexs) , np.nanstd(Cindexs)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')
        info(f'Overall test Cindex = {cimean_score:.6f} +/- {cistd_score:.6f} ')
        if args.show_individual_scores:
            for task_num, task_name in enumerate(args.task_names):
                info(f'\tOverall test {task_name} {metric} = '
                     f'{np.nanmean(scores[:, task_num]):.6f} +/- {np.nanstd(scores[:, task_num]):.6f}')

    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)

        header = ['Task']
        for metric in args.metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(args.num_folds)]
        writer.writerow(header)

        for task_num, task_name in enumerate(args.task_names):
            row = [task_name]
            for metric, scores in all_scores.items():
                task_scores = scores[:, task_num]
                mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)
        writer.writerow(['Cindex', cimean_score, cistd_score])

    avg_scores = np.nanmean(all_scores[args.metric], axis=1)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)

    if args.save_preds:
        all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', 'test_preds.csv'))
                               for fold_num in range(args.num_folds)])
        all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return mean_score, std_score


def chemprop_train() -> None:
    cross_validate(args=TrainArgs().parse_args(), train_func=run_training)
