import os, pickle, random

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch3d.transforms import random_rotation
import math

from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph, to_dense_batch, to_dense_adj

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from einops import rearrange
import time
import sidechainnet as scn

import torch.utils
from sidechainnet.utils.sequence import VOCAB, DSSPVocabulary

from collections import defaultdict
from functools import partial

import torch.nn.functional as F

from glob import glob
from tqdm import tqdm

from mp_nerf.mp_nerf_utils import ensure_chirality
from copy import deepcopy

TRAIN_DATASETS = ['DIPS']
VALIDATION_DATASETS = [] #'valid-10', 'valid-70', 'valid-30'] # 'valid-10',  'valid-70' 'valid-90'] # 'valid-20', 'valid-30', 'valid-40', 'valid-50',
TEST_DATASETS = ['test']


DATASETS = TRAIN_DATASETS + VALIDATION_DATASETS + TEST_DATASETS

DSSP_VOCAV = DSSPVocabulary()

import torch.nn.functional as F

def rot_matrix(a, b, c):
    a1, a2 = a - b, c - b
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

res_to_mirror_symmetry = {
    "D": 1,
    "F": 1,
    "Y": 1,
    "E": 2
}

def get_alternative_angles(seq, angles):
    ambiguous = res_to_mirror_symmetry.keys()
    angles = deepcopy(angles)
    for res_idx, res in enumerate(seq):
        if res in ambiguous:
            angles[res_idx][6+res_to_mirror_symmetry[res]] -= math.pi
    return angles

def create_dataloaders(config):
    data = {}
    data['DIPS'] = scn.load(local_scn_path=os.path.join(config.dataset_source, 'dips_800_pruned.pkl'))
    loaders = {
        split_name: ProteinComplexDataset(
           data[split_name],
           split_name=split_name,
           dataset_source=config.dataset_source,
           downsample=config.downsample,
           max_seq_len=config.max_seq_len,
        ).make_loader(config.batch_size, config.num_workers)
        for split_name in ('DIPS', )
    }
    return loaders

def load_embedding(name, source, type='seq'):
    encodings_dir = os.path.join(source, f'{type}_encodings')
    filepath = os.path.join(encodings_dir, name + '.pyd')
    if not os.path.exists(filepath): return (None, None)
    with open(filepath, 'rb') as f:
        emb, att = pickle.load(f)
    return emb, att

class ProteinComplexDataset(torch.utils.data.Dataset):
    def __init__(self,
                 scn_data_split,
                 split_name,
                 dataset_source,
                 seq_clamp=184,
                 downsample=1.0,
                 max_seq_len=256,
                 add_sos_eos=False,
                 sort_by_length=False,
                 reverse_sort=True):

        print(f'\n=============== Loading {split_name}!')
        num_proteins = len(scn_data_split['seq'])
        self.max_seq_len = max_seq_len

        # shuffle
        indices = list(range(num_proteins))
        random.shuffle(indices)
        for key, attribute in scn_data_split.items():
            scn_data_split[key] = list(map(attribute.__getitem__, indices))

        # for training, we consider the downsample flag
        # and prune the dataset by protein size
        if split_name in TRAIN_DATASETS or split_name in VALIDATION_DATASETS:
            random_filter = random.sample(range(num_proteins), int(num_proteins*downsample))
            for key, attribute in scn_data_split.items():
                scn_data_split[key] = list(map(attribute.__getitem__, random_filter))

            length_filter = [idx for idx, seq in enumerate(scn_data_split['seq'])
                                                if len(seq) < self.max_seq_len]
            for key, attribute in scn_data_split.items():
                scn_data_split[key] = list(map(attribute.__getitem__, length_filter))

        # standard SCN approach to handling data
        self.seqs = [VOCAB.str2ints(s, add_sos_eos) for s in scn_data_split['seq']]
        self.str_seqs = scn_data_split['seq']
        self.angs = scn_data_split['ang']
        self.crds = scn_data_split['crd']
        self.tgt_crds = scn_data_split['tgt_crd'] if 'tgt_crd' in scn_data_split else None
        self.tgt_angs = scn_data_split['tgt_ang'] if 'tgt_ang' in scn_data_split else None
        self.chns = scn_data_split['chn']
        self.ids = scn_data_split['ids']
        self.resolutions = scn_data_split['res']
        self.secs = [DSSP_VOCAV.str2ints(s, add_sos_eos) for s in scn_data_split['sec']]
        self.split_name = split_name
        self.seq_clamp = seq_clamp
        print(f'=============== {split_name} was loaded!\n')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        new_idx = random.randint(0, len(self)-1)
        seqs = self.seqs[idx]
        num_nodes = len(seqs)

        chains = torch.LongTensor(self.chns[idx])
        chains = rearrange(chains, '(r a) -> r a', a=14)[:, 1]
        if random.random() < 0.5: chains = (chains + 1) % 2
        chains = chains.float() + 1

        crds = torch.FloatTensor(self.crds[idx]).reshape(-1, 14, 3)
        rots = rot_matrix(crds[:, 0], crds[:, 1], crds[:, 2])

        tgt_crds = torch.FloatTensor(self.tgt_crds[idx]).clone() if self.tgt_crds else crds.clone()
        tgt_rots = rot_matrix(tgt_crds[:, 0], tgt_crds[:, 1], tgt_crds[:, 2])

        backbone_coords = crds[:, 1, :]
        distance_map = torch.cdist(backbone_coords, backbone_coords)

        edge_index = torch.nonzero(torch.ones(num_nodes, num_nodes)).t()

        v, u = edge_index
        edge_vectors = torch.einsum('b p, b p q -> b q', backbone_coords[u] - backbone_coords[v],
                                                         rots[v].transpose(-1, -2))
        edge_angles = F.normalize(edge_vectors, dim=-1)

        subunit1 = crds[chains == 1] - crds[chains == 1].mean(dim=-3)
        crds[chains == 1] = torch.einsum('b a p, p q -> b a q', subunit1, random_rotation().t())

        subunit2 = crds[chains == 2] - crds[chains == 2].mean(dim=-3)
        crds[chains == 2] = torch.einsum('b a p, p q -> b a q', subunit2, random_rotation().t())
        crds[chains == 2] = crds[chains == 2] # + torch.FloatTensor([50, 50, 50])[None, :]
        backbone_coords = crds[:, 1, :]

        datum = Data(
            __num_nodes__=len(seqs),
            ids=self.ids[idx],
            crds=crds,
            tgt_crds=tgt_crds,
            bck_crds=backbone_coords,
            rots=rots,
            chns=chains,
            seqs=torch.LongTensor(seqs) + 1,
            angs=torch.FloatTensor(self.angs[idx]),
            str_seqs=self.str_seqs[idx],
            edge_index=edge_index,
            edge_distance=distance_map[v, u],
            edge_vectors=edge_vectors,
            edge_angles=edge_angles
        )

        datum = deepcopy(datum)
        if (datum.crds[:, 1, :].norm(dim=-1).gt(1e-6) & datum.crds[:, 1, 0].isfinite()).sum() < 10:
            return self[new_idx]

        return datum

    def __str__(self):
        """Describe this dataset to the user."""
        return (f"ProteinDataset( "
                f"split='{self.split_name}', "
                f"n_proteins={len(self)}, ")

    def __repr__(self):
        return self.__str__()

    def make_loader(self, batch_size, num_workers, data_structure='batch'):
        loader = DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        return loader


def collate_fn(stream):
    batch = Batch.from_data_list(stream)
    for key in ('seqs', 'tgt_crds', 'crds', 'bck_crds', 'angs', 'rots', 'chns'):
        if batch[key] is None: continue
        batch[key], mask = to_dense_batch(batch[key], batch=batch.batch)
    for key in ('edge_distance', 'edge_angles'):
        if batch[key] is None: continue
        batch[key] = to_dense_adj(edge_index=batch.edge_index,
                                batch=batch.batch, edge_attr=batch[key])

    # used for both models and losses
    batch.node_pad_mask = mask
    batch.edge_pad_mask = (mask[:, None, :] & mask[:, :, None])

    # used for losses
    batch.node_record_mask = batch.crds[:, :, 1].norm(dim=-1).gt(1e-6) & batch.crds[:, :, 1, 0].isfinite()
    batch.angle_record_mask = batch.angs.ne(0.0) & batch.angs.isfinite()
    batch.edge_record_mask = batch.edge_distance.gt(0) & batch.edge_angles.sum(-1).ne(0)

    return batch
