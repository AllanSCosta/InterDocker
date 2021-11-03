
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


def process_complex(complex_tuple):
    complex_path, complex = complex_tuple

    chains = []
    destination = os.path.join(complex_path, 'scn_complex.pkl')
    if os.path.exists(destination):
        print(f'{destination} already converted!')
        return

    for i, chain in enumerate(('chain_1.pdb', 'chain_2.pdb')):
        chain = os.path.join(complex_path, chain)
        chain = pr.parsePDB(chain)
        if not chain:
            print(f'{complex_path} has no chain!')
            return

        try:
            dihedrals, coords, sequence, unmodified_seq, is_nonstd = get_seq_coords_and_angles(chain)
        except:
            print(f'{complex_path} failed to properly parse')
            return

        resolution = 0
        dssp = " " * len(sequence)
        data = {
            "ang": dihedrals,
            "crd": coords,
            "seq": sequence,
            "chn": np.full([len(coords)], i, dtype=int)
        }
        chains.append(data)

    subunit1, subunit2 = chains
    data = {
        'ids': complex,
        'ang': np.concatenate((subunit1['ang'], subunit2['ang']), axis=0),
        'crd': np.concatenate((subunit1['crd'], subunit2['crd']), axis=0),
        'seq': subunit1['seq'] + subunit2['seq'],
        'chn': np.concatenate((subunit1['chn'], subunit2['chn']), axis=0)
    }

    with open(destination, 'wb') as file:
        pickle.dump(data, file)

    print(f'{destination} just converted!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to embed sequences')

    parser.add_argument('--source', type=str, default='../../data/dips_preprocessed/')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_workers', type=int, default=40)

    args = parser.parse_args()
    source = os.path.join(args.source, args.split)

    paths = []
    for section in os.listdir(source):
        section_path = os.path.join(source, section)
        for complex in os.listdir(section_path):
            paths.append((os.path.join(section_path, complex), complex))

    process_map(process_complex, paths, max_workers=args.max_workers)
