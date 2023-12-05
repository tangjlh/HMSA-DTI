import json
from logging import Logger
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from HMSA.args import TrainArgs
from HMSA.constants import MODEL_FILE_NAME
from HMSA.data import get_class_sizes, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from HMSA.models import HMSAModel
from HMSA.utils import build_lr_scheduler, get_loss_func, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model
from .lamb import Lamb
from lifelines.utils import concordance_index

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run_training(args: TrainArgs,
                 data: MoleculeDataset,
                 fold: int,
                 logger: Logger = None,
                 tokenizer=None
                 ) -> Dict[str, List[float]]:

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    torch.manual_seed(args.pytorch_seed)

    debug(f'Splitting data with seed {args.seed}')
    train_data, val_data, test_data = split_data(data=data,
                                                 fold=fold,
                                                 split_type=args.split_type,
                                                 sizes=args.split_sizes,
                                                 seed=args.seed,
                                                 num_folds=args.num_folds,
                                                 args=args,
                                                 logger=logger
                                                 )

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger,
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    if args.bond_feature_scaling and args.bond_features_size > 0:
        bond_feature_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_features=True)
        val_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
        test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
    else:
        bond_feature_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(train_data)+len(val_data)+len(test_data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    scaler = None
    loss_func = get_loss_func(args)

    test_smiles, test_sequences, test_targets = test_data.smiles(), test_data.sequences(), test_data.targets()

    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )

    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = HMSAModel(args)

        if args.checkpoint_frzn is not None:
            debug(f'Loading and freezing parameters from {args.checkpoint_frzn}.')
            model = load_frzn_model(model=model,path=args.checkpoint_frzn, current_args=args, logger=logger)

        model = model.cuda()

        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
                        features_scaler, atom_descriptor_scaler, bond_feature_scaler, args)

        optimizer = Lamb(model.parameters(), lr=args.lamp_lr, weight_decay=0.01, betas=(.9, .999), adam=True)
        scheduler = build_lr_scheduler(optimizer, args)
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer,
                tokenizer= tokenizer
            )

            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                args=args,
                scaler=scaler,
                logger=logger,
                tokenizer=tokenizer
            )

            for metric, scores in val_scores.items():
                avg_val_score = np.nanmean(scores)
                debug(f'Validation {metric} = {avg_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', avg_val_score, n_iter)

                if args.show_individual_scores:
                    for task_name, val_score in zip(args.task_names, scores):
                        debug(f'Validation {task_name} {metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)

            avg_val_score = np.nanmean(val_scores[args.metric])
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler,
                                atom_descriptor_scaler, bond_feature_scaler, args)

        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        test_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler,
            tokenizer=tokenizer,
            args=args
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            logger=logger,
            test=True
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{metric}', avg_test_score, 0)

            if args.show_individual_scores:
                for task_name, test_score in zip(args.task_names, scores):
                    info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                    writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
        writer.close()

    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    print('ensemble learning')
    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metrics=args.metrics,
        dataset_type=args.dataset_type,
        logger=logger,
    )
    prediction = []
    label = []
    for i in range(args.num_tasks):
        for j in range(len(avg_test_preds)):
            if test_targets[j][i] is not None:
                prediction.append(avg_test_preds[j][i])
                label.append(float(test_targets[j][i]))
                
    cindex = concordance_index(label,prediction)
    for metric, scores in ensemble_scores.items():
        avg_ensemble_test_score = np.nanmean(scores)
        info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')

        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, scores):
                info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    with open(os.path.join(args.save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

    if args.save_preds:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles_save()})

        for i, task_name in enumerate(args.task_names):
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores, cindex
