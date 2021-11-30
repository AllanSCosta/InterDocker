import os, pickle, random

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch3d.transforms import random_rotation
import math

from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph, to_dense_batch, to_dense_adj

from scipy.spatial import KDTree
from einops import repeat

from einops import rearrange
import time
import sidechainnet as scn

import torch.utils
from sidechainnet.utils.sequence import VOCAB, DSSPVocabulary

from collections import defaultdict
from functools import partial

import torch.nn.functional as F

from mp_nerf.mp_nerf_utils import ensure_chirality

from copy import deepcopy

TRAIN_DATASETS = ['train']
TEST_DATASETS = ['DB5']
VALIDATION_DATASETS = ['val']

DATASETS = TRAIN_DATASETS + VALIDATION_DATASETS + TEST_DATASETS

DSSP_VOCAV = DSSPVocabulary()


def rot_matrix(a, b, c):
    a1, a2 = a - b, c - b
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def create_dataloaders(config):
    print(f'Loading DIPS - {"sample" if config.debug else "full"}')
    filename = f'dips_1024_pruned_esm_128{"_sample" if config.debug else ""}.pkl'
    filepath = os.path.join(config.dataset_source, filename)

    with open(filepath, 'rb') as file:
        start = time.time()
        data = pickle.load(file)
        print(f'{time.time() - start:.2f} seconds to load dataset')

    loaders = {
        split: ProteinComplexDataset(
           dataset,
           split_name=split,
           dataset_source=config.dataset_source,
           spatial_clamp=config.spatial_clamp,
           downsample=config.downsample,
           max_seq_len=config.max_seq_len,
        ).make_loader(config.batch_size if split not in TEST_DATASETS else 1, config.num_workers)
        for split, dataset in data.items()
    }

    return loaders


class ProteinComplexDataset(torch.utils.data.Dataset):
    def __init__(self,
                 scn_data_split,
                 split_name,
                 dataset_source,
                 spatial_clamp=128,
                 downsample=1.0,
                 max_seq_len=1000000,
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

            length_filter = [idx for idx, seq in enumerate(scn_data_split['seq']) if len(seq) < self.max_seq_len]
            for key, attribute in scn_data_split.items():
                scn_data_split[key] = list(map(attribute.__getitem__, length_filter))

        # standard SCN approach to handling data
        self.str_seqs = scn_data_split['seq']
        self.angs = scn_data_split['ang']
        self.crds = scn_data_split['crd']
        self.chns = scn_data_split['chn']
        self.encs = scn_data_split['enc']
        self.ids = scn_data_split['ids']

        if 'tgt_crd' in scn_data_split:
            self.tgt_crds = scn_data_split['tgt_crd']

        self.split_name = split_name
        self.spatial_clamp = spatial_clamp
        print(f'=============== {split_name} was loaded! {split_name} has size {len(self.str_seqs)} \n')


    def __len__(self):
        return len(self.str_seqs)


    def __getitem__(self, idx):
        new_idx = random.randint(0, len(self)-1)

        datum = ProteinComplexDataset.build_datum(
            self.ids[idx], self.str_seqs[idx],
            self.crds[idx], self.angs[idx],
            self.chns[idx], encs=self.encs[idx], tgt_crds=self.tgt_crds[idx] if hasattr(self, 'tgt_crds') else None
        )

        datum = deepcopy(datum)
        if self.spatial_clamp > 0 and datum.__num_nodes__ > self.spatial_clamp and self.split_name not in TEST_DATASETS:
            datum = self.kd_clamp(datum)

        if (datum.crds[:, 1, :].norm(dim=-1).gt(1e-6) & datum.crds[:, 1, 0].isfinite()).sum() < 10:
            return self[new_idx]

        datum = self.homomeric_augmentation(datum)
        chains = self.split_chains(datum)

        return chains

    @classmethod
    def build_datum(cls, ids, str_seqs, crds, angs, chns, encs=None, tgt_crds=None):
        seqs = VOCAB.str2ints(str_seqs, False)
        num_nodes = len(seqs)

        chains = torch.LongTensor(chns)
        if len(chains.size()) == 2:
            chains = chains[:, 1]
        if random.random() < 0.5: chains = (chains + 1) % 2
        chains = chains.float() + 1

        has_bound = (tgt_crds is not None)
        tgt_crds = tgt_crds if has_bound else crds
        tgt_crds = torch.FloatTensor(tgt_crds).reshape(-1, 14, 3)
        tgt_crds = ensure_chirality(tgt_crds.unsqueeze(0)).squeeze(0)
        tgt_rots = rot_matrix(tgt_crds[:, 0], tgt_crds[:, 1], tgt_crds[:, 2])
        crds = torch.FloatTensor(crds).reshape(-1, 14, 3) if has_bound else tgt_crds.clone()
        tgt_bck_coords = tgt_crds[:, 1, :]

        distance_map = torch.cdist(tgt_bck_coords, tgt_bck_coords)
        edge_index = torch.nonzero(torch.ones(num_nodes, num_nodes)).t()

        v, u = edge_index
        edge_vectors = torch.einsum('b p, b p q -> b q', tgt_bck_coords[u] - tgt_bck_coords[v],
                                                         tgt_rots[v].transpose(-1, -2))
        edge_angles = F.normalize(edge_vectors, dim=-1)
        # move chains to origin and random rotate
        subunit1 = crds[chains == 1] - crds[chains == 1].mean(dim=-3)
        crds[chains == 1] = torch.einsum('b a p, p q -> b a q', subunit1, random_rotation().t())
        subunit2 = crds[chains == 2] - crds[chains == 2].mean(dim=-3)
        crds[chains == 2] = torch.einsum('b a p, p q -> b a q', subunit2, random_rotation().t())

        rots = rot_matrix(crds[:, 0], crds[:, 1], crds[:, 2])
        bck_coords = crds[:, 1, :]

        datum = Data(
            num_nodes=len(seqs),
            __num_nodes__=len(seqs),
            ids=ids,
            crds=crds,
            tgt_crds=tgt_bck_coords,
            tgt_rots=tgt_rots,
            bck_crds=bck_coords,
            rots=rots,
            chns=chains,
            seqs=torch.LongTensor(seqs) + 1,
            encs=torch.FloatTensor(encs.type(torch.float32)),
            angs=torch.FloatTensor(angs),
            str_seqs=str_seqs,
            edge_index=edge_index,
            edge_distance=distance_map[v, u],
            edge_vectors=edge_vectors,
            edge_angles=edge_angles
        )

        return datum

    def split_chains(self, datum):
        chains = []
        for chain in (1, 2):
            nodes_filter = datum.chns == chain

            chain_subgraph = Data()
            chain_subgraph.crds = datum.crds[nodes_filter]
            chain_subgraph.tgt_crds = datum.tgt_crds[nodes_filter]
            chain_subgraph.tgt_rots = datum.tgt_rots[nodes_filter]
            chain_subgraph.bck_crds=datum.bck_crds[nodes_filter]
            chain_subgraph.rots=datum.rots[nodes_filter]
            chain_subgraph.chns=datum.chns[nodes_filter]
            chain_subgraph.seqs=datum.seqs[nodes_filter]
            chain_subgraph.angs=datum.angs[nodes_filter]
            chain_subgraph.encs=datum.encs[nodes_filter]
            chain_subgraph.str_seqs=[chr for chr, f in zip(datum.str_seqs, nodes_filter) if f]
            chain_subgraph.ids = datum.ids

            edge_index, chain_subgraph.edge_distance = subgraph(nodes_filter, datum.edge_index,
                        edge_attr=datum.edge_distance, relabel_nodes=True)
            _, chain_subgraph.edge_vectors = subgraph(nodes_filter, datum.edge_index,
                        edge_attr=datum.edge_vectors, relabel_nodes=True)
            _, chain_subgraph.edge_angles = subgraph(nodes_filter, datum.edge_index,
                        edge_attr=datum.edge_angles, relabel_nodes=True)

            chain_subgraph.edge_index = edge_index
            chain_subgraph.__num_nodes__ = chain_subgraph.num_nodes = nodes_filter.sum().item()
            chains.append(chain_subgraph)

        return chains


    def kd_clamp(self, datum):
        v, u = datum.edge_index
        is_external = datum.chns[v] != datum.chns[u]
        contacts = datum.edge_index[:, (datum.edge_distance < 12) & is_external].T
        sample_idx = random.sample(range(len(contacts)), k=1)[0]
        sample_contact = contacts[sample_idx]

        nodes_filter = torch.zeros_like(datum.chns).type(torch.bool)
        for anchor in sample_contact:
            anchor_chain, anchor_crd = datum.chns[anchor], datum.bck_crds[anchor]
            chain_mask = datum.chns != anchor_chain
            chain_crds = datum.bck_crds + chain_mask.float()[:, None] * torch.full_like(datum.bck_crds, 1000)

            tree = KDTree(chain_crds)
            _, neighbors = tree.query(anchor_crd, k=min(self.spatial_clamp, chain_mask.sum().item()))
            neighbors = torch.LongTensor(neighbors)
            nodes_filter[neighbors] = True

        datum.crds = datum.crds[nodes_filter]
        datum.tgt_crds = datum.tgt_crds[nodes_filter]
        datum.tgt_rots = datum.tgt_rots[nodes_filter]
        datum.bck_crds=datum.bck_crds[nodes_filter]
        datum.rots=datum.rots[nodes_filter]
        datum.chns=datum.chns[nodes_filter]
        datum.seqs=datum.seqs[nodes_filter]
        datum.angs=datum.angs[nodes_filter]
        datum.encs=datum.encs[nodes_filter]
        datum.str_seqs=[chr for chr, f in zip(datum.str_seqs, nodes_filter) if f]

        edge_index, datum.edge_distance = subgraph(nodes_filter, datum.edge_index,
                    edge_attr=datum.edge_distance, relabel_nodes=True)
        _, datum.edge_vectors = subgraph(nodes_filter, datum.edge_index,
                    edge_attr=datum.edge_vectors, relabel_nodes=True)
        _, datum.edge_angles = subgraph(nodes_filter, datum.edge_index,
                    edge_attr=datum.edge_angles, relabel_nodes=True)

        datum.edge_index = edge_index
        datum.__num_nodes__ = datum.num_nodes = nodes_filter.sum().item()

        return datum


    def homomeric_augmentation(self, datum):
        # note that symmetry is barely broken when n, m ~= 128 but n, m < 128
        # homomeric encoding happens after the cut
        chain1, chain2 = datum.chns == 1, datum.chns == 2
        is_homomeric = ((chain1.sum().item() == chain2.sum().item()) and
                        torch.all(datum.seqs[chain1] == datum.seqs[chain2]))
        datum.homo = torch.BoolTensor([is_homomeric])

        if is_homomeric:
            first, last = datum.chns[0], datum.chns[-1]
            original_coords = datum.tgt_crds
            alternate_crds = original_coords[datum.chns == last], original_coords[datum.chns == first]
            alternate_crds = torch.cat(alternate_crds, dim=0)
            alternate_distance_map = torch.cdist(alternate_crds, alternate_crds)
            alternate_edge_distance = alternate_distance_map[datum.edge_index[0], datum.edge_index[1]]

            alternate_rots = (datum.tgt_rots[datum.chns == last], datum.tgt_rots[datum.chns == last])
            alternate_rots = torch.cat(alternate_rots, dim=0)

            datum.edge_distance = torch.stack((datum.edge_distance, alternate_edge_distance), dim=-1)

            datum.tgt_crds = torch.stack((datum.tgt_crds, alternate_crds), dim=-1)
            datum.tgt_rots = torch.stack((datum.tgt_rots, alternate_rots), dim=-1)

            v, u = datum.edge_index
            alternate_edge_vectors = torch.einsum('b p, b p q -> b q', alternate_crds[u] - alternate_crds[v],
                                                             alternate_rots[v].transpose(-1, -2))
            alternate_edge_angles = F.normalize(alternate_edge_vectors, dim=-1)
            datum.edge_angles = torch.stack((datum.edge_angles, alternate_edge_angles), dim=-2)

        else:
            datum.edge_distance = torch.stack((datum.edge_distance, datum.edge_distance), dim=-1)
            datum.tgt_crds = torch.stack((datum.tgt_crds, datum.tgt_crds), dim=-1)
            datum.tgt_rots = torch.stack((datum.tgt_rots, datum.tgt_rots), dim=-1)
            datum.edge_angles = torch.stack((datum.edge_angles, datum.edge_angles), dim=-2)

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
    batches = [Batch.from_data_list(list(s)) for s in zip(*stream)]
    for batch in batches:
        for key in ('seqs', 'tgt_crds', 'tgt_rots', 'crds', 'bck_crds', 'angs', 'rots', 'chns', 'encs'):
            if key not in batch or batch[key] is None: continue
            batch[key], mask = to_dense_batch(batch[key], batch=batch.batch)
        for key in ('edge_distance', 'edge_angles', 'edge_vectors'):
            if key not in batch or batch[key] is None: continue
            batch[key] = to_dense_adj(edge_index=batch.edge_index,
                                    batch=batch.batch, edge_attr=batch[key])

        # used for both models and losses
        batch.node_pad_mask = mask
        batch.edge_pad_mask = (mask[:, None, :] & mask[:, :, None])

        # used for losses
        batch.node_record_mask = batch.crds[:, :, 1].norm(dim=-1).gt(1e-6) & batch.crds[:, :, 1, 0].isfinite()
        batch.angle_record_mask = batch.angs.ne(0.0) & batch.angs.isfinite()
        batch.edge_record_mask = batch.edge_distance[..., 0].gt(0) & batch.edge_angles[..., 0, :].sum(-1).ne(0)

    n, m = [p.encs.size(1) for p in batches]
    endpoints = repeat(batches[0].tgt_crds, 'b n p s -> b n m p s', m=m), repeat(batches[1].tgt_crds, 'b m c s -> b n m c s', n=n)
    edge_vectors = torch.einsum('b n m p s, b n p q s -> b n m q s', endpoints[0] - endpoints[1], batches[0].tgt_rots.transpose(-2, -3))

    return (*batches, edge_vectors)
