from collections import defaultdict
import logging
from typing import Dict, List
from sklearn.metrics import (accuracy_score, precision_score, precision_recall_curve, recall_score)
from .predict import predict
from HMSA.data import MoleculeDataLoader, StandardScaler
from HMSA.models import HMSAModel
from HMSA.utils import get_metric_func
from HMSA.args import TrainArgs
import os


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metrics: List[str],
                         dataset_type: str,
                         logger: logging.Logger = None,
                         test: bool = False) -> Dict[str, List[float]]:

    info = logger.info if logger is not None else print

    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] * num_tasks for metric in metrics}

    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    valid_pred_label = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_pred_label[i].append(1 if preds[j][i] >=0.5 else 0)
                valid_targets[i].append(targets[j][i])

    prec = precision_score(valid_targets[0], valid_pred_label[0])
    recall = recall_score(valid_targets[0], valid_pred_label[0])
    acc = accuracy_score(valid_targets[0], valid_pred_label[0])
    tpr, fpr, _ = precision_recall_curve(valid_targets[0], valid_pred_label[0])
    if test:
        with open(os.getcwd() + '/' + 'pred_target.txt', 'a') as f:
            f.write(str(preds)+'\n')
            f.write(str(targets)+'\n\n')
        print(f'test_acc={acc:.5f}' + '\n' +
              f'test_recall={recall:.5f}' + '\n' +
              f'test_prec={prec:.5f}' + '\n'
              )
        with open(os.getcwd() + '/' + 'result.txt', 'a') as f:
            f.write('test_acc=' + str(acc) + '\n')
            f.write('test_recall=' + str(recall) + '\n')
            f.write('test_prec=' + str(prec) + '\n')

    else:
        print(f'valid_acc={acc:.5f}' + '\n' +
              f'valid_recall={recall:.5f}' + '\n' +
              f'valid_prec={prec:.5f}' + '\n'
              )
    results = defaultdict(list)
    for i in range(num_tasks):
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                for metric in metrics:
                    results[metric].append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        for metric, metric_func in metric_to_func.items():
            results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

    results = dict(results)

    return results


def evaluate(model: HMSAModel,
             data_loader: MoleculeDataLoader,
             num_tasks: int,
             metrics: List[str],
             dataset_type: str,
             args:TrainArgs,
             scaler: StandardScaler = None,
             logger: logging.Logger = None, tokenizer = None) -> Dict[str, List[float]]:

    preds = predict(
        model=model,
        data_loader=data_loader,
        args=args,
        scaler=scaler,
        tokenizer=tokenizer
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        num_tasks=num_tasks,
        metrics=metrics,
        dataset_type=dataset_type,
        logger=logger
    )

    return results
