from argparse import Namespace
import csv
from datetime import timedelta
from functools import wraps
import logging
import math
import os
import pickle
import re
from time import time
from typing import Any, Callable, List, Tuple, Union
import collections
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from HMSA.args import PredictArgs, TrainArgs
from HMSA.data import StandardScaler, MoleculeDataset, preprocess_smiles_columns, get_task_names
from HMSA.models import HMSAModel
from HMSA.nn_utils import NoamLR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def makedirs(path: str, isfile: bool = False) -> None:

    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str,
                    model: HMSAModel,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None,
                    atom_descriptor_scaler: StandardScaler = None,
                    bond_feature_scaler: StandardScaler = None,
                    args: TrainArgs = None) -> None:
    if args is not None:
        args = Namespace(**args.as_dict())

    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None,
        'atom_descriptor_scaler': {
            'means': atom_descriptor_scaler.means,
            'stds': atom_descriptor_scaler.stds
        } if atom_descriptor_scaler is not None else None,
        'bond_feature_scaler': {
            'means': bond_feature_scaler.means,
            'stds': bond_feature_scaler.stds
        } if bond_feature_scaler is not None else None
    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    device: torch.device = None,
                    logger: logging.Logger = None) -> HMSAModel:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    args.from_dict(vars(state['args']), skip_unsettable=True)
    loaded_state_dict = state['state_dict']

    if device is not None:
        args.device = device

    model = HMSAModel(args)
    model_state_dict = model.state_dict()

    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        if re.match(r'(encoder\.encoder\.)([Wc])', loaded_param_name):
            param_name = loaded_param_name.replace('encoder.encoder', 'encoder.encoder.0')
        else:
            param_name = loaded_param_name

        if param_name not in model_state_dict:
            info(f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            info(f'Warning: Pretrained parameter "{loaded_param_name}" '
                 f'of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding '
                 f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug('Moving model to cuda')
    model = model.cuda()

    return model


def overwrite_state_dict(loaded_param_name: str,
                        model_param_name: str,
                        loaded_state_dict: collections.OrderedDict,
                        model_state_dict: collections.OrderedDict,
                        logger: logging.Logger = None) -> collections.OrderedDict:
    debug = logger.debug if logger is not None else print

    
    if model_param_name not in model_state_dict:
        debug(f'Pretrained parameter "{model_param_name}" cannot be found in model parameters.')
        
    elif model_state_dict[model_param_name].shape != loaded_state_dict[loaded_param_name].shape:
        debug(f'Pretrained parameter "{loaded_param_name}" '
              f'of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding '
              f'model parameter of shape {model_state_dict[model_param_name].shape}.')
    
    else:
        debug(f'Loading pretrained parameter "{model_param_name}".')
        model_state_dict[model_param_name] = loaded_state_dict[loaded_param_name]    
    
    return model_state_dict


def load_frzn_model(model: torch.nn,
                    path: str,
                    current_args: Namespace = None,
                    cuda: bool = None,
                    logger: logging.Logger = None) -> HMSAModel:

    debug = logger.debug if logger is not None else print

    loaded_mpnn_model = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = loaded_mpnn_model['state_dict']
    loaded_args = loaded_mpnn_model['args']

    model_state_dict = model.state_dict()
    
    if loaded_args.number_of_molecules==1 & current_args.number_of_molecules==1:      
        encoder_param_names = ['encoder.encoder.0.W_i.weight', 'encoder.encoder.0.W_h.weight', 'encoder.encoder.0.W_o.weight', 'encoder.encoder.0.W_o.bias']
        if current_args.checkpoint_frzn is not None:
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)
            
        if current_args.frzn_ffn_layers > 0:         
            ffn_param_names = [['ffn.'+str(i*3+1)+'.weight','ffn.'+str(i*3+1)+'.bias'] for i in range(current_args.frzn_ffn_layers)]
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]
            
            for param_name in encoder_param_names+ffn_param_names:
                model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)               
            
        if current_args.freeze_first_only:
            debug(f'WARNING: --freeze_first_only flag cannot be used with number_of_molecules=1 (flag is ignored)')
            
    elif (loaded_args.number_of_molecules==1) & (current_args.number_of_molecules>1):
        
        if (current_args.checkpoint_frzn is not None) & (current_args.freeze_first_only) & (not (current_args.frzn_ffn_layers > 0)):
            encoder_param_names = ['encoder.encoder.0.W_i.weight', 'encoder.encoder.0.W_h.weight', 'encoder.encoder.0.W_o.weight', 'encoder.encoder.0.W_o.bias']
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)
                
        if (current_args.checkpoint_frzn is not None) & (not current_args.freeze_first_only) & (not (current_args.frzn_ffn_layers > 0)):
            loaded_encoder_param_names = ['encoder.encoder.0.W_i.weight', 'encoder.encoder.0.W_h.weight', 'encoder.encoder.0.W_o.weight', 'encoder.encoder.0.W_o.bias']*current_args.number_of_molecules
            model_encoder_param_names = [['encoder.encoder.'+str(mol_num)+'.W_i.weight', 'encoder.encoder.'+str(mol_num)+'.W_h.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.bias'] for mol_num in range(current_args.number_of_molecules)]
            model_encoder_param_names = [item for sublist in model_encoder_param_names for item in sublist]
            for loaded_param_name,model_param_name in zip(loaded_encoder_param_names,model_encoder_param_names):
                model_state_dict = overwrite_state_dict(loaded_param_name,model_param_name,loaded_state_dict,model_state_dict)
        
        if current_args.frzn_ffn_layers > 0:
            raise Exception ('Number of molecules in checkpoint_frzn must be equal to current model for ffn layers to be frozen')
            
    elif (loaded_args.number_of_molecules>1 )& (current_args.number_of_molecules>1):
        if (loaded_args.number_of_molecules) !=( current_args.number_of_molecules):
            raise Exception('Number of molecules in checkpoint_frzn ({}) must match current model ({}) OR equal to 1.'.format(loaded_args.number_of_molecules,current_args.number_of_molecules))
        
        if current_args.freeze_first_only:
            raise Exception('Number of molecules in checkpoint_frzn ({}) must be equal to 1 for freeze_first_only to be used.'.format(loaded_args.number_of_molecules))
       
        if (current_args.checkpoint_frzn is not None) & (not (current_args.frzn_ffn_layers > 0)):
            encoder_param_names = [['encoder.encoder.'+str(mol_num)+'.W_i.weight', 'encoder.encoder.'+str(mol_num)+'.W_h.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.bias'] for mol_num in range(current_args.number_of_molecules)]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]
            
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)
        
        if current_args.frzn_ffn_layers > 0:
                
            encoder_param_names = [['encoder.encoder.'+str(mol_num)+'.W_i.weight', 'encoder.encoder.'+str(mol_num)+'.W_h.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.bias'] for mol_num in range(current_args.number_of_molecules)]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]            
            ffn_param_names = [['ffn.'+str(i*3+1)+'.weight','ffn.'+str(i*3+1)+'.bias'] for i in range(current_args.frzn_ffn_layers)]
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]
            
            for param_name in encoder_param_names + ffn_param_names:
                model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)    
        
        if current_args.frzn_ffn_layers >= current_args.ffn_num_layers:
            raise Exception('Number of frozen FFN layers must be less than the number of FFN layers')

    model.load_state_dict(model_state_dict)
    
    return model


def load_scalers(path: str) -> Tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler]:

    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    if 'atom_descriptor_scaler' in state.keys():
        atom_descriptor_scaler = StandardScaler(state['atom_descriptor_scaler']['means'],
                                                state['atom_descriptor_scaler']['stds'],
                                                replace_nan_token=0) if state['atom_descriptor_scaler'] is not None else None
    else:
        atom_descriptor_scaler = None

    if 'bond_feature_scaler' in state.keys():
        bond_feature_scaler = StandardScaler(state['bond_feature_scaler']['means'],
                                            state['bond_feature_scaler']['stds'],
                                            replace_nan_token=0) if state['bond_feature_scaler'] is not None else None
    else:
        bond_feature_scaler = None

    return scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler


def load_args(path: str) -> TrainArgs:
    args = TrainArgs()
    args.from_dict(vars(torch.load(path, map_location=lambda storage, loc: storage)['args']), skip_unsettable=True)

    return args


def load_task_names(path: str) -> List[str]:
    return load_args(path).task_names


def get_loss_func(args: TrainArgs) -> nn.Module:
    if args.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def bce(targets: List[int], preds: List[float]) -> float:
    bce_func = nn.BCELoss(reduction='mean')
    loss = bce_func(target=torch.Tensor(targets), input=torch.Tensor(preds)).item()
    return loss


def rmse(targets: List[float], preds: List[float]) -> float:
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets: List[float], preds: List[float]) -> float:
    return mean_squared_error(targets, preds)


def accuracy(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    if type(preds[0]) == list:
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]
    return accuracy_score(targets, hard_preds)


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    if metric == 'auc':
        return roc_auc_score
    if metric == 'prc-auc':
        return prc_auc
    if metric == 'rmse':
        return rmse
    if metric == 'mse':
        return mse
    if metric == 'mae':
        return mean_absolute_error
    if metric == 'r2':
        return r2_score
    if metric == 'accuracy':
        return accuracy
    if metric == 'cross_entropy':
        return log_loss
    if metric == 'binary_cross_entropy':
        return bce
    raise ValueError(f'Metric "{metric}" not supported.')


def build_optimizer(model: nn.Module, args: TrainArgs) -> Optimizer:
    params = [{'params': model.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]

    return Adam(params)


def build_lr_scheduler(optimizer: Optimizer, args: TrainArgs, total_epochs: List[int] = None):
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr]
    )


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    def timeit_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f'Elapsed time = {delta}')
            return result
        return wrap
    return timeit_decorator


def save_smiles_splits(data_path: str,
                       save_dir: str,
                       task_names: List[str] = None,
                       features_path: List[str] = None,
                       train_data: MoleculeDataset = None,
                       val_data: MoleculeDataset = None,
                       test_data: MoleculeDataset = None,
                       logger: logging.Logger = None,
                       smiles_columns: List[str] = None) -> None:
    makedirs(save_dir)
    
    info = logger.info if logger is not None else print
    save_split_indices = True

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=data_path, smiles_columns=smiles_columns)

    with open(data_path) as f:
        reader = csv.DictReader(f)

        indices_by_smiles = {}
        for i, row in enumerate(tqdm(reader)):
            smiles = tuple([row[column] for column in smiles_columns])
            if smiles in indices_by_smiles:
                save_split_indices = False
                info('Warning: Repeated SMILES found in data, pickle file of split indices cannot distinguish entries and will not be generated.')
                break
            indices_by_smiles[smiles] = i

    if task_names is None:
        task_names = get_task_names(path=data_path, smiles_columns=smiles_columns)

    features_header = []
    if features_path is not None:
        for feat_path in features_path:
            with open(feat_path, 'r') as f:
                reader = csv.reader(f)
                feat_header = next(reader)
                features_header.extend(feat_header)

    all_split_indices = []
    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f'{name}_smiles.csv'), 'w') as f:
            writer = csv.writer(f)
            if smiles_columns[0] == '':
                writer.writerow(['smiles'])
            else:
                writer.writerow(smiles_columns)
            for smiles in dataset.smiles():
                writer.writerow(smiles)

        with open(os.path.join(save_dir, f'{name}_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(smiles_columns + task_names)
            dataset_targets = dataset.targets()
            for i, smiles in enumerate(dataset.smiles()):
                writer.writerow(smiles + dataset_targets[i])

        if features_path is not None:
            dataset_features = dataset.features()
            with open(os.path.join(save_dir, f'{name}_features.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(features_header)
                writer.writerows(dataset_features)

        if save_split_indices:
            split_indices = []
            for smiles in dataset.smiles():
                index = indices_by_smiles.get(tuple(smiles))
                if index is None:
                    save_split_indices = False
                    info(f'Warning: SMILES string in {name} could not be found in data file, and likely came from a secondary data file. '
                    'The pickle file of split indices can only indicate indices for a single file and will not be generated.')
                    break
                split_indices.append(index)
            else:
                split_indices.sort()
                all_split_indices.append(split_indices)

        if name == 'train':
            data_weights = dataset.data_weights()
            if any([w != 1 for w in data_weights]):
                with open(os.path.join(save_dir, f'{name}_weights.csv'),'w') as f:
                    writer=csv.writer(f)
                    writer.writerow(['data weights'])
                    for weight in data_weights:
                        writer.writerow([weight])

    if save_split_indices:
        with open(os.path.join(save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)


def update_prediction_args(predict_args: PredictArgs,
                           train_args: TrainArgs,
                           missing_to_defaults: bool = True,
                           validate_feature_sources: bool = True) -> None:

    for key, value in vars(train_args).items():
        if not hasattr(predict_args, key):
            setattr(predict_args, key, value)

    if missing_to_defaults:
        override_defaults = {
            'bond_features_scaling':False,
            'no_bond_features_scaling':True,
            'atom_descriptors_scaling':False,
            'no_atom_descriptors_scaling':True,
        }
        default_train_args=TrainArgs().parse_args(['--data_path', None, '--dataset_type', str(train_args.dataset_type)])
        for key, value in vars(default_train_args).items():
            if not hasattr(predict_args,key):
                setattr(predict_args,key,override_defaults.get(key,value))
    
    if train_args.number_of_molecules != predict_args.number_of_molecules:
        raise ValueError('A different number of molecules was used in training '
                        f'model than is specified for prediction, {train_args.number_of_molecules} '
                         'smiles fields must be provided')

    if train_args.atom_descriptors != predict_args.atom_descriptors:
        raise ValueError('The use of atom descriptors is inconsistent between training and prediction. If atom descriptors '
                         ' were used during training, they must be specified again during prediction using the same type of '
                         ' descriptors as before. If they were not used during training, they cannot be specified during prediction.')

    if (train_args.bond_features_path is None) != (predict_args.bond_features_path is None):
        raise ValueError('The use of bond descriptors is different between training and prediction. If you used bond '
                         'descriptors for training, please specify a path to new bond descriptors for prediction.')

    if train_args.features_scaling != predict_args.features_scaling:
        raise ValueError('If scaling of the additional features was done during training, the '
                         'same must be done during prediction.')

    if train_args.atom_descriptors != predict_args.atom_descriptors:
        raise ValueError('The use of atom descriptors is inconsistent between training and prediction. '
                         'If atom descriptors were used during training, they must be specified again '
                         'during prediction using the same type of descriptors as before. '
                         'If they were not used during training, they cannot be specified during prediction.')

    if (train_args.bond_features_path is None) != (predict_args.bond_features_path is None):
        raise ValueError('The use of bond descriptors is different between training and prediction. If you used bond'
                         'descriptors for training, please specify a path to new bond descriptors for prediction.')

    if validate_feature_sources:
        if (((train_args.features_path is None) != (predict_args.features_path is None))
            or ((train_args.features_generator is None) != (predict_args.features_generator is None))):
            raise ValueError('Features were used during training so they must be specified again during prediction '
                            'using the same type of features as before (with either --features_generator or '
                            '--features_path and using --no_features_scaling if applicable).')
