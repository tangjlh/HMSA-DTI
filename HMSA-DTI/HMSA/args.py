import json
import os
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple
from typing_extensions import Literal
import torch
from tap import Tap
import HMSA.data.utils
from HMSA.data import set_cache_mol, empty_cache
from HMSA.features import get_available_features_generators

Metric = Literal['auc', 'prc-auc', 'r2', 'accuracy', 'cross_entropy', 'binary_cross_entropy']


def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.pt') -> Optional[List[str]]:

    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError('Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths')

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')

        return checkpoint_paths

    return None


class CommonArgs(Tap):

    smiles_columns: List[str] = None
    number_of_molecules: int = 1
    checkpoint_dir: str = None
    checkpoint_path: str = None
    checkpoint_paths: List[str] = None
    no_cuda: bool = False
    gpu: int = None
    features_generator: List[str] = None
    features_path: List[str] = None
    no_features_scaling: bool = False
    max_data_size: int = None
    num_workers: int = 8
    batch_size: int = 50
    atom_descriptors: Literal['feature', 'descriptor'] = None
    atom_descriptors_path: str = None
    bond_features_path: str = None
    no_cache_mol: bool = False
    empty_cache: bool = False

    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0

    @property
    def device(self) -> torch.device:
        if not self.cuda:
            return torch.device('cpu')
        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        return not self.no_features_scaling

    @features_scaling.setter
    def features_scaling(self, features_scaling: bool) -> None:
        self.no_features_scaling = not features_scaling

    @property
    def atom_features_size(self) -> int:
        return self._atom_features_size

    @atom_features_size.setter
    def atom_features_size(self, atom_features_size: int) -> None:
        self._atom_features_size = atom_features_size

    @property
    def atom_descriptors_size(self) -> int:
        return self._atom_descriptors_size

    @atom_descriptors_size.setter
    def atom_descriptors_size(self, atom_descriptors_size: int) -> None:
        self._atom_descriptors_size = atom_descriptors_size

    @property
    def bond_features_size(self) -> int:
        return self._bond_features_size

    @bond_features_size.setter
    def bond_features_size(self, bond_features_size: int) -> None:
        self._bond_features_size = bond_features_size

    def configure(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        self.add_argument('--features_generator', choices=get_available_features_generators())

    def process_args(self) -> None:
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
        )

        if self.features_generator is not None and 'rdkit_2d_normalized' in self.features_generator and self.features_scaling:
            raise ValueError('When using rdkit_2d_normalized features, --no_features_scaling must be specified.')

        if (self.atom_descriptors is None) != (self.atom_descriptors_path is None):
            raise ValueError('If atom_descriptors is specified, then an atom_descriptors_path must be provided '
                             'and vice versa.')

        if self.atom_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        if self.bond_features_path is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Bond descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        set_cache_mol(not self.no_cache_mol)

        if self.empty_cache:
            empty_cache()


class TrainArgs(CommonArgs):

    data_path: str
    target_columns: List[str] = None
    ignore_columns: List[str] = None
    dataset_type: str = 'classification'
    separate_val_path: str = None
    separate_test_path: str = None
    data_weights_path: str = None
    target_weights: List[float] = None
    split_type: str = 'random'
    split_sizes: Tuple[float, float, float] = (0.81, 0.09, 0.10)
    num_folds: int = 1
    folds_file: str = None
    val_fold_index: int = None
    test_fold_index: int = None
    crossval_index_dir: str = None
    crossval_index_file: str = None
    seed: int = 0
    pytorch_seed: int = 0
    metric: Metric = None
    extra_metrics: List[Metric] = []
    save_dir: str = None
    checkpoint_frzn: str = None
    save_smiles_splits: bool = False
    test: bool = False
    quiet: bool = False
    log_frequency: int = 10
    show_individual_scores: bool = False
    cache_cutoff: float = 10000
    save_preds: bool = False
    resume_experiment: bool = False
    bias: bool = False
    hidden_size: int = 300
    depth: int = 3
    dmpnn_shared: bool = False
    dropout: float = 0.1
    activation: str = 'ReLU'
    atom_messages: bool = False
    undirected: bool = False
    ffn_hidden_size: int = None
    ffn_num_layers: int = 2
    features_only: bool = False
    separate_val_features_path: List[str] = None
    separate_test_features_path: List[str] = None
    separate_val_atom_descriptors_path: str = None
    separate_test_atom_descriptors_path: str = None
    separate_val_bond_features_path: str = None
    separate_test_bond_features_path: str = None
    config_path: str = None
    ensemble_size: int = 1
    aggregation: Literal['mean', 'sum', 'norm'] = 'mean'
    aggregation_norm: int = 100
    reaction: bool = False
    reaction_mode: Literal['reac_prod', 'reac_diff', 'prod_diff'] = 'reac_diff'

    explicit_h: bool = False
    epochs: int = 30
    warmup_epochs: float = 2.0
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4
    grad_clip: float = None
    class_balance: bool = False
    overwrite_default_atom_features: bool = False
    no_atom_descriptor_scaling: bool = False
    overwrite_default_bond_features: bool = False
    no_bond_features_scaling: bool = False
    frzn_ffn_layers: int = 0
    freeze_first_only: bool = False
    sequence_length: int = 500
    lamp_lr: float = 0.0025
    tau: Tuple[float, float] = (0.0, 5.0)
    alpha: float = 0.5
    beta: float = 5
    vocab_size: int = 31
    two_mer_classes: int = 485
    three_mer_classes: int = 10649
    one_mer_classes: int = 23
    smiles_element_classes: int = 65
    prot_hidden: int = 300
    prot_1d_out: int = 64
    prot_1dcnn_num: int = 3
    kernel_size: int = 7

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def metrics(self) -> List[str]:
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy', 'binary_cross_entropy'}

    @property
    def use_input_features(self) -> bool:
        return self.features_generator is not None or self.features_path is not None

    @property
    def num_lrs(self) -> int:
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    @property
    def atom_descriptor_scaling(self) -> bool:
        return not self.no_atom_descriptor_scaling

    @property
    def bond_feature_scaling(self) -> bool:
        return not self.no_bond_features_scaling

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        global temp_dir
        self.smiles_columns = HMSA.data.utils.preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)
        if self.save_dir is None:
            temp_dir = TemporaryDirectory()
            self.save_dir = temp_dir.name
        if self.checkpoint_paths is not None and len(self.checkpoint_paths) > 0:
            self.ensemble_size = len(self.checkpoint_paths)
        if self.metric is None:
            if self.dataset_type == 'classification':
                self.metric = 'auc'
        if self.metric in self.extra_metrics:
            raise ValueError(f'Metric {self.metric} is both the metric and is in extra_metrics. '
                             f'Please only include it once.')

        for metric in self.metrics:
            if not ((self.dataset_type == 'classification' and metric in ['auc', 'prc-auc', 'accuracy', 'binary_cross_entropy'])):
                raise ValueError(f'Metric "{metric}" invalid for dataset type "{self.dataset_type}".')

        if self.class_balance and self.dataset_type != 'classification':
            raise ValueError('Class balance can only be applied if the dataset type is classification.')

        if self.features_only and not (self.features_generator or self.features_path):
            raise ValueError('When using features_only, a features_generator or features_path must be provided.')

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size

        if self.atom_messages and self.undirected:
            raise ValueError('Undirected is unnecessary when using atom_messages '
                             'since atom_messages are by their nature undirected.')

        if self.test:
            self.epochs = 0

        if self.separate_val_path is not None and self.atom_descriptors is not None \
                and self.separate_val_atom_descriptors_path is None:
            raise ValueError('Atom descriptors are required for the separate validation set.')

        if self.separate_test_path is not None and self.atom_descriptors is not None \
                and self.separate_test_atom_descriptors_path is None:
            raise ValueError('Atom descriptors are required for the separate test set.')

        if self.separate_val_path is not None and self.bond_features_path is not None \
                and self.separate_val_bond_features_path is None:
            raise ValueError('Bond descriptors are required for the separate validation set.')

        if self.separate_test_path is not None and self.bond_features_path is not None \
                and self.separate_test_bond_features_path is None:
            raise ValueError('Bond descriptors are required for the separate test set.')

        if self.overwrite_default_atom_features and self.atom_descriptors != 'feature':
            raise NotImplementedError('Overwriting of the default atom descriptors can only be used if the'
                                      'provided atom descriptors are features.')

        if not self.atom_descriptor_scaling and self.atom_descriptors is None:
            raise ValueError('Atom descriptor scaling is only possible if additional atom features are provided.')

        if self.overwrite_default_bond_features and self.bond_features_path is None:
            raise ValueError('If you want to overwrite the default bond descriptors, '
                             'a bond_descriptor_path must be provided.')

        if not self.bond_feature_scaling and self.bond_features_path is None:
            raise ValueError('Bond descriptor scaling is only possible if additional bond features are provided.')

        if self.target_weights is not None:
            avg_weight = sum(self.target_weights) / len(self.target_weights)
            self.target_weights = [w / avg_weight for w in self.target_weights]
            if min(self.target_weights) < 0:
                raise ValueError('Provided target weights must be non-negative.')


class PredictArgs(CommonArgs):
    test_path: str
    preds_path: str
    drop_extra_columns: bool = False
    ensemble_variance: bool = False

    sequence_length: int = 500

    lamp_lr: float = 0.0025

    tau: Tuple[float, float] = (0.0, 5.0)

    alpha: float = 0.5

    beta: float = 5

    vocab_size: int = 31

    prot_hidden: int = 300

    prot_1d_out: int = 64

    prot_1dcnn_num: int = 3

    kernel_size: int = 7

    @property
    def ensemble_size(self) -> int:
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        super(PredictArgs, self).process_args()

        self.smiles_columns = HMSA.data.utils.preprocess_smiles_columns(
            path=self.test_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')


class InterpretArgs(CommonArgs):

    data_path: str
    batch_size: int = 50
    property_id: int = 1
    rollout: int = 20
    c_puct: float = 10.0
    max_atoms: int = 20
    min_atoms: int = 8
    prop_delta: float = 0.5

    def process_args(self) -> None:
        super(InterpretArgs, self).process_args()

        self.smiles_columns = HMSA.data.utils.preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        if self.features_path is not None:
            raise ValueError('Cannot use --features_path <path> for interpretation since features '
                             'need to be computed dynamically for molecular substructures. '
                             'Please specify --features_generator <generator>.')

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')


class HyperoptArgs(TrainArgs):

    num_iters: int = 20
    config_save_path: str
    log_dir: str = None


class SklearnTrainArgs(TrainArgs):

    model_type: Literal['random_forest', 'svm']
    class_weight: Literal['balanced'] = None
    single_task: bool = False
    radius: int = 2
    num_bits: int = 2048
    num_trees: int = 500


class SklearnPredictArgs(Tap):

    test_path: str
    smiles_columns: List[str] = None

    number_of_molecules: int = 1

    preds_path: str
    checkpoint_dir: str = None
    checkpoint_path: str = None
    checkpoint_paths: List[str] = None

    def process_args(self) -> None:
        self.smiles_columns = HMSA.data.utils.preprocess_smiles_columns(
            path=self.test_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
            ext='.pkl'
        )
