import scipy.spatial as spa
import numpy as np
import pandas as pd
import torch
import os
import time

from torch import nn

import argparse
import atom3d.datasets as da

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Data, Dataset, Batch

import pickle
from tqdm import tqdm
import esm
import pickle
import random
import wandb
os.environ["WANDB_MODE"] = "dryrun"

import matplotlib.pyplot as plt
import esm

from functools import partial
from collections import defaultdict


def augment_dataset(path, funnel, transformer, batch_converter, dtype=torch.float16):
    source = os.path.join(path, 'scn_complex.pkl')
    if not os.path.exists(source):
        print(f'{source} datum doesnt exist')
        return

    destination = os.path.join(path, f'scn_complex_esm_{funnel.weight.shape[0]}.pkl')

    # if os.path.exists(destination):
        # print(f'{destination} already computed')
        # return

    with open(source, 'rb') as file:
        complex = pickle.load(file)

    assert len(complex['chn'].shape) != 2
    cut = int(len(complex['chn']) - complex['chn'].sum())

    print(cut, len(complex['chn']))
    residues1, residues2 = complex['seq'][:cut], complex['seq'][cut:]

    if len(residues1) >= 1022 or len(residues2) >= 1022: return

    _, __, tokens = batch_converter([('A', residues1), ('B', residues2)])
    inference = transformer(tokens.cuda(), repr_layers=[33])
    embedding1, embedding2 = inference['representations'][33].cpu()

    embedding1 = embedding1[1:len(residues1)+1]
    embedding2 = embedding2[1:len(residues2)+1]
    encoding = torch.cat((embedding1, embedding2), dim=0)
    encoding = funnel(encoding).type(dtype)

    complex['enc'] = encoding
    with open(destination, 'wb') as file:
        pickle.dump(complex, file)

    print(f'{destination} just computed')


def submit_map(params):
    params = dict(vars(params))
    script_path = os.path.realpath(__file__)
    base_path = os.getcwd()

    # make logging workspace
    for rank in range(params['world_size']):
        worskpace_dir = os.path.join(base_path, 'workspace')
        os.makedirs(worskpace_dir, exist_ok=True)

        # put our call bash in there
        script = os.path.join(worskpace_dir, f'worker_{rank}.sh')

        # build preamble
        preamble = f'#!/bin/bash\n'
        preamble += f'#SBATCH --gres=gpu:volta:1\n'
        preamble += f'#SBATCH -o {os.path.join(worskpace_dir, "worker_" + str(rank))}.sh.log\n'
        preamble += f'#SBATCH --cpus-per-task=20\n'
        preamble += f'#SBATCH --job-name=worker_{rank}\n\n'
        preamble += f'module load anaconda/2021a\n'

        # parse parameters
        params_list = [(key, value) for key, value in params.items() if (key != 'rank' and key != 'submit_map')]
        params_list.append(('rank', rank))
        params_strings = [f'--{key} {str(value) if type(value) != list else " ".join([str(v) for v in value])}' for key, value in params_list]
        params_string = ' '.join(params_strings)

        # call file
        call = f'python -u {script_path} {params_string}'

        # build script
        with open(script, 'w') as file:
            file.write(preamble + call)

        # submit
        os.system(f'LLsub {script}')
        print(f'submitted worker #{rank}')

        # sleep a bit not to push too much to the scheduler
        time.sleep(5)

    print(f'submitted {args.world_size}! jobs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to embed sequences')

    parser.add_argument('--source', type=str, default='../../data/dips_preprocessed/')
    parser.add_argument('--split', type=str, default='DB5')
    parser.add_argument('--pca', type=int, default=128)

    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=1)

    # if submit_map is set, rank is ignored and a pool of {1...world_size} workers is generated
    parser.add_argument('--submit_map', dest='submit_map', action='store_true')

    args = parser.parse_args()

    if args.submit_map:
        submit_map(args)
        exit()

    source = os.path.join(args.source, args.split)

    print('Loading ESM')
    esm1b, esm1b_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm1b = esm1b.eval().cuda()
    esm1b_batch_converter = esm1b_alphabet.get_batch_converter()

    if args.pca > 0:
        assert args.pca in [64, 128, 256, 512]
        with open(f'lm_pca_components/pca_comp_{args.pca}.pyd', 'rb') as file:
            pca_comp = pickle.load(file)
            print(f'producing final embeddings with a {pca_comp.shape} linear transformation')

        funnel = nn.Linear(*pca_comp.T.shape, bias=False)
        funnel.weight.shape
        with torch.no_grad():
            funnel.weight = nn.Parameter(torch.FloatTensor(pca_comp))

    augmenter = partial(
        augment_dataset,
        funnel=funnel,
        transformer=esm1b,
        batch_converter=esm1b_batch_converter
    )

    paths = []
    for section in os.listdir(source):
        section_path = os.path.join(source, section)
        for complex in os.listdir(section_path):
            paths.append(os.path.join(section_path, complex))

    print(f'detected {len(paths)} complexes to embed')
    if args.world_size > 1:
        chunks = np.array_split(paths, args.world_size)
        paths = chunks[args.rank]

    with torch.no_grad():
        for path in tqdm(paths):
            augmenter(path)
