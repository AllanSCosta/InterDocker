import pickle
import os
import argparse

from collections import defaultdict
from functools import partial

import torch
from torch import nn
import numpy as np
from einops import rearrange

from mp_nerf.protein_utils import build_scaffolds_from_scn_angles
from mp_nerf.proteins import sidechain_fold, ca_bb_fold
from sidechainnet import StructureBuilder
from sidechainnet.utils.measure import get_seq_coords_and_angles
from prody import parsePDB
import esm

from visualization import plot_timeseries, plot_predictions
from data import ProteinComplexDataset, collate_fn
from model import Interactoformer
from utils import logit_expectation

from datetime import datetime


def load_pdbs(path1, path2):
    chains = defaultdict(list)
    for chain_idx, path in enumerate([path1, path2]):
        pdb = parsePDB(path)
        dihedrals, coords, sequence, _, _ = get_seq_coords_and_angles(pdb)
        chain_labels = torch.full([len(coords)], chain_idx, dtype=int)
        chains['ang'].append(torch.FloatTensor(dihedrals))
        chains['crd'].append(torch.FloatTensor(coords))
        chains['chn'].append(torch.LongTensor(chain_labels))
        chains['seq'].append(sequence)

    (seq1, seq2) = chains['seq']
    chains['seq'] = seq1 + seq2
    chains['id'] = os.path.basename(args.p1) + '_' + os.path.basename(args.p1)
    chains['ang'] = torch.cat(chains['ang'], dim=0)
    chains['crd'] = torch.cat(chains['crd'], dim=0)
    chains['chn'] = torch.cat(chains['chn'], dim=0)

    return chains, (seq1, seq2)


if __name__ == '__main__':
    with torch.no_grad():
        torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    parser.add_argument('--p1', type=str, default='/home/gridsan/allanc/DNAInteract_shared/allan/sample_pdb/1mcz_A.pdb')
    parser.add_argument('--p2', type=str, default='/home/gridsan/allanc/DNAInteract_shared/allan/sample_pdb/1mcz_B.pdb')
    parser.add_argument('--model_path', type=str, default='/home/gridsan/allanc/DNAInteract_shared/allan/models/bipartite-portico')

    parser.add_argument('--out_path', type=str, default='./inference/')

    parser.add_argument('--cross_encoder_steps', type=int, default=20)
    parser.add_argument('--docker_steps', type=int, default=30)

    parser.add_argument('--logits', type=int, default=0)
    parser.add_argument('--timeseries', type=int, default=1)
    parser.add_argument('--pdb', type=int, default=0)
    parser.add_argument('--image', type=int, default=0)

    args = parser.parse_args()

    config_path = os.path.join(args.model_path, 'config.pkl')
    weights_path = os.path.join(args.model_path, 'checkpoint.pt')

    datum, (seq1, seq2) = load_pdbs(args.p1, args.p2)
    if len(seq1) >= 1022 or len(seq2) >= 1022:
        print('ESM-1b only accepts sequences of length <= 1022')
        exit()

    with open(config_path, 'rb') as file:
        config = pickle.load(file)

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Loading InterDocker: {config.name}')
    interdocker = Interactoformer(config)
    interdocker.eval_steps = args.docker_steps
    interdocker.load_state_dict(torch.load(weights_path))
    interdocker = interdocker.eval().to(device)

    print('Loading ESM-1b')
    esm1b, esm1b_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm1b = esm1b.eval().to(device)
    esm1b_batch_converter = esm1b_alphabet.get_batch_converter()

    with open(f'data_processing/lm_pca_components/pca_comp_128.pyd', 'rb') as file:
        pca_comp = pickle.load(file)
        print(f'producing final embeddings with a {pca_comp.shape} linear transformation')

    funnel = nn.Linear(*pca_comp.T.shape, bias=False)
    with torch.no_grad():
        funnel.weight = nn.Parameter(torch.FloatTensor(pca_comp))

    _, __, tokens = esm1b_batch_converter([('A', seq1), ('B', seq2)])
    inference = esm1b(tokens.cuda(), repr_layers=[33])
    embedding1, embedding2 = inference['representations'][33].cpu()

    embedding1 = embedding1[1:len(seq1)+1]
    embedding2 = embedding2[1:len(seq2)+1]
    encoding = torch.cat((embedding1, embedding2), dim=0)
    encoding = funnel(encoding)
    datum['enc'] = encoding

    datum = ProteinComplexDataset.build_datum(
        datum['id'], datum['seq'], datum['crd'],
        datum['ang'], datum['chn'], datum['enc']
    )

    batch = collate_fn([datum]).to(device)
    output = interdocker(batch, is_training=False)

    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y-%H-%M-%S")

    folder_name = f'{datum.ids}_{interdocker.config.name}_{timestamp}'
    out_folder = os.path.join(args.out_path, folder_name)
    os.makedirs(out_folder, exist_ok=True)

    if args.timeseries:
        print('Plotting Timeseries')
        timeseries = rearrange(output['translations'], 'b i t c -> t b i c')
        view = plot_timeseries(datum.str_seqs, datum.angs.cpu(), timeseries.cpu(), (datum.chns == datum.chns[0]).sum().item())

        timeseries_path = os.path.join(out_folder, 'timeseries.html')
        with open(timeseries_path, 'w') as file:
            file.write(view._make_html())

    if args.pdb:
        final_coords = timeseries[-1].cpu()
        backbone = ca_bb_fold(final_coords)[0]
        scaffolds = build_scaffolds_from_scn_angles(datum.str_seqs, angles=datum.angs.cpu(), device="cpu")
        coords, _ = sidechain_fold(wrapper = backbone.clone(), **scaffolds, c_beta = 'torsion')
        full_atom = StructureBuilder(datum.str_seqs, coords.reshape(-1, 3))

        pdb_path = os.path.join(out_folder, 'docked.pdb')
        with open(pdb_path, 'w') as file:
            file.write(full_atom.to_pdbstr())

    logits = output['logit_traj']
    log_output = dict()

    distance_logits = rearrange(logits['distance'], 'b l ... -> l b ...')[-1]
    log_output['dist'] = distance_logits

    distance_expectation = logit_expectation(distance_logits)
    images = [ distance_expectation ]

    if 'angles' in logits:
        angle_logits = rearrange(logits['angles'], 'b l ... (a e) -> l a b ... e', a=3)[-1]
        log_output['ang'] = angle_logits
        for angle_idx in range(3):
            angle_expectation = logit_expectation(angle_logits[angle_idx])
            images.append(angle_expectation)

    if args.logits:
        logit_path = os.path.join(out_folder, 'logits.pyd')
        with open(logit_path, 'wb') as file:
            pickle.dump(log_output, file)

    if args.image:
        external_edges = (rearrange(batch.chns, 'b s -> b () s') != rearrange(batch.chns, 'b s -> b s ()'))
        images = [img[external_edges] for img in images]
        images = [img[:int(len(img)/2)] for img in images]
        images = [rearrange(img, '(n m) -> n m', n=len(seq1), m=len(seq2)).cpu() for img in images]
        images = [img.detach().cpu() for img in images]
        fig = plot_predictions(images)

        img_path = os.path.join(out_folder, 'distograms.png')
        fig.savefig(img_path)
