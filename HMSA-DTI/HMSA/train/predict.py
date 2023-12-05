from typing import List
import torch
from tqdm import tqdm
from HMSA.data import MoleculeDataLoader, StandardScaler
from HMSA.models import HMSAModel
import numpy as np
from HMSA.args import TrainArgs


def predict(model: HMSAModel,
            data_loader: MoleculeDataLoader,
            args: TrainArgs,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            tokenizer=None) -> List[List[float]]:

    model.eval()
    preds = []
    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        mol_batch, target_batch, protein_sequence_batch, data_weights_batch, kmer, smiles =\
            batch.batch_graph(), batch.targets(), batch.sequences(), batch.data_weights(), batch.two_mer(), batch.smiles()
        dummy_array = [0] * args.sequence_length
        sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
        new_ar = []
        for arr in sequence_2_ar:
            while len(arr)>args.sequence_length:
                arr.pop(len(arr)-1)
            new_ar.append(np.zeros(args.sequence_length)+np.array(arr))

        sequence_tensor = torch.LongTensor(np.array(new_ar))

        with torch.no_grad():
            batch_preds = model(mol_batch, kmer, smiles, sequence_tensor)
        batch_preds = batch_preds.data.cpu().numpy()

        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
    return preds
