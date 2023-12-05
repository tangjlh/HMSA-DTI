import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim import lr_scheduler as _LRScheduler
from tqdm import tqdm
from HMSA.args import TrainArgs
from HMSA.data import MoleculeDataLoader
from HMSA.models import HMSAModel
import numpy as np


def train(model: HMSAModel,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          tokenizer=None) -> int:

    model.train()
    loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        mol_batch, target_batch, protein_sequence_batch, data_weights_batch, kmer, smiles =\
            batch.batch_graph(), batch.targets(), batch.sequences(), batch.data_weights(), batch.two_mer(), batch.smiles()

        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        mask_weight = torch.Tensor(
            [[args.alpha if list(args.tau)[0] <= x <= list(args.tau)[1] else args.beta for x in tb] for tb in
             target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
        if args.target_weights is not None:
            target_weights = torch.Tensor(args.target_weights)
        else:
            target_weights = torch.ones_like(targets)
        data_weights = torch.Tensor(data_weights_batch).unsqueeze(1)

        model.zero_grad()
        dummy_array = [0] * args.sequence_length

        sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
        new_ar = []

        for arr in sequence_2_ar:
            while len(arr) > args.sequence_length:
                arr.pop(len(arr) - 1)

            new_ar.append(np.zeros(args.sequence_length) + np.array(arr))

        sequence_tensor = torch.LongTensor(np.array(new_ar))
        preds = model(mol_batch, kmer, smiles, sequence_tensor)

        mask = mask.to(preds.device)
        mask_weight = mask_weight.to(preds.device)
        targets = targets.to(preds.device)

        target_weights = target_weights.to(preds.device)
        data_weights = data_weights.to(preds.device)

        loss = loss_func(preds, targets) * target_weights * data_weights * mask_weight
        loss = loss.sum() / mask.sum()
        loss_sum += loss.item()
        iter_count += 1
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n_iter += len(batch)
    return n_iter