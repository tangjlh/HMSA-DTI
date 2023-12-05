import os
from typing import List, Union, Tuple
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
from .dmpnn import DMPNN
from HMSA.args import TrainArgs
from HMSA.features import BatchMolGraph
from HMSA.nn_utils import initialize_weights

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class HMSAModel(nn.Module):

    def __init__(self, args: TrainArgs):

        super(HMSAModel, self).__init__()
        self.classification = args.dataset_type == 'classification'
        self.embedding_xt = nn.Embedding(args.vocab_size, args.prot_hidden*2)
        self.attention = nn.MultiheadAttention(args.hidden_size, 1)
        self.fc1 = nn.Linear(args.hidden_size * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)
        self.d3 = nn.Dropout(p=0.1)
        self.leaky = nn.LeakyReLU()
        self.embedding_1mer = nn.Embedding(args.one_mer_classes, args.prot_hidden * 2)
        self.embedding_2mer = nn.Embedding(args.two_mer_classes, args.prot_hidden * 2)
        self.embedding_3mer = nn.Embedding(args.three_mer_classes, args.prot_hidden * 2)
        self.CNN_kmer = nn.Sequential(
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=args.hidden_size, kernel_size=16),
            nn.ReLU()
        )
        self.embedding_smiles = nn.Embedding(args.smiles_element_classes, args.prot_hidden * 2)
        self.CNN_smiles = nn.Sequential(
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=args.hidden_size, kernel_size=4),
            nn.ReLU()
        )
        self.CNN_SEQ = nn.Sequential(
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=args.hidden_size, kernel_size=4),
            nn.ReLU()
        )
        self.ffn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=4),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=4)
        )
        if self.classification:
            self.sigmoid = nn.Sigmoid()

        self.create_encoder(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs):
        self.encoder = DMPNN(args)
        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def forward(self,
                batch: Union[
                    List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                kmer_tensor: torch.Tensor = None,
                smiles_tensor: torch.Tensor = None,
                sequence_tensor: List[np.ndarray] = None
                ):

        mpnn_out = self.encoder(batch)

        smiles_tensor = smiles_tensor.cuda()
        embedding_smiles = self.embedding_smiles(smiles_tensor).permute(0, 2, 1)
        smiles_in = self.CNN_smiles(embedding_smiles).permute(0, 2, 1)

        kmer_tensor = kmer_tensor.cuda()
        embedding_kmer = self.embedding_3mer(kmer_tensor).permute(0, 2, 1)
        kmer_in = self.CNN_kmer(embedding_kmer).permute(0, 2, 1)

        sequence_tensor = sequence_tensor.cuda()
        embedded_xt = self.embedding_xt(sequence_tensor).permute(0,2,1)
        out_conv = self.CNN_SEQ(embedded_xt)
        protein_tensor = out_conv.permute(0, 2, 1)

        drug = torch.cat((mpnn_out, smiles_in), dim=1).permute(1, 0, 2)
        drug_att, _ = self.attention(drug, drug, drug)

        protein = torch.cat((kmer_in, protein_tensor), dim=1).permute(1, 0, 2)
        protein_att, _ = self.attention(protein, protein, protein)

        drug_pro = torch.cat((drug_att, protein_att), dim=0)
        drug_pro, _ = self.attention(drug_pro, drug_pro, drug_pro)

        drug_mpnn, drug_smiles, protein_kmer, protein_seq = drug_pro[:mpnn_out.size(1), :, :].permute(1, 0, 2), \
            drug_pro[mpnn_out.size(1):mpnn_out.size(1) + smiles_in.size(1), :, ].permute(1, 0, 2), \
            drug_pro[mpnn_out.size(1) + smiles_in.size(1):drug_pro.size(0) - protein_tensor.size(1), :, ].permute(1, 0,
                                                                                                                  2), \
            drug_pro[drug_pro.size(0) - protein_tensor.size(1):, :, ].permute(1, 0, 2)

        drug_smiles_max_pool = nn.MaxPool1d(smiles_in.size(1))
        drug_mpnn_max_pool = nn.MaxPool1d(mpnn_out.size(1))
        protein_seq_max_pool = nn.MaxPool1d(protein_tensor.size(1))
        protein_kmer_max_pool = nn.MaxPool1d(kmer_in.size(1))
        drug_mpnn_flatten = drug_mpnn_max_pool((0.5 * drug_mpnn + 0.5 * mpnn_out).permute(0, 2, 1))
        drug_smiles_flatten = drug_smiles_max_pool((0.5 * drug_smiles + 0.5 * smiles_in).permute(0, 2, 1))
        protein_kmer_flatten = protein_kmer_max_pool((0.5 * protein_kmer + 0.5 * kmer_in).permute(0, 2, 1))
        protein_seq_flatten = protein_seq_max_pool((0.5 * protein_seq + 0.5 * protein_tensor).permute(0, 2, 1))

        pair = torch.cat((drug_mpnn_flatten.squeeze(2), drug_smiles_flatten.squeeze(2), protein_kmer_flatten.squeeze(2),
                          protein_seq_flatten.squeeze(2)), dim=1)
        f1 = self.d2(self.leaky(self.fc1(self.d1(pair))))
        f2 = self.d3(self.leaky(self.fc2(f1)))
        f3 = self.leaky(self.fc3(f2))
        output = self.fc4(f3)
        return output
