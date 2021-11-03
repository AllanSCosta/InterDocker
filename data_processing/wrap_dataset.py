
import os
import prody as pr
import sidechainnet as scn
from sidechainnet.utils.measure import get_seq_coords_and_angles
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import pickle
import argparse

from tqdm.contrib.concurrent import process_map

import re

def load_complex(path_tuple):
    path, complex = path_tuple
    if not os.path.exists(path): return None

    subunits = [(m.start(0), m.end(0)) for m in re.finditer('[A-Z]+_[0-10]*', complex)][0]
    sub_names = complex[:subunits[0]+1], complex[subunits[1]:]
    target_name = f'{sub_names[0][:4] + sub_names[0][-1]}_{sub_names[1][:4] + sub_names[1][-1]}'
    path = f'../data/{target_name[:2]}/{target_name}'

    try:
        with open(os.path.join('../data', target_name), 'rb') as file:
            encodings = pickle.load(file)
        with open(os.path.join(path, 'scn_complex.pkl'), 'rb') as file:
            complex = pickle.load(file)
    except:
        return None

    return complex

if __name__ == '__main__':
    pdb_path = '/home/gridsan/allanc/language_models_benchmarks/dips_pdb'
    dataset = defaultdict(list)

    paths = []
    for section in os.listdir(pdb_path):
        section_path = os.path.join(pdb_path, section)
        for complex in os.listdir(section_path):
            paths.append((os.path.join(section_path, complex), complex))

    complexes = []
    for path in paths:
        complexes.append(load_complex(path))
    # complexes = process_map(load_complex, paths)
    # dataset = defaultdict(list)

    for complex in tqdm(complexes):
        if not complex: continue
        for key in ('ids', 'crd', 'ang', 'seq', 'chn'):
            dataset[key].append(complex[key])

    with open('dips_800_pruned.pkl', 'wb') as file:
        pickle.dump(dataset, file)
    print(f'final dataset has size {len(dataset["seq"])}')
    print('done')
