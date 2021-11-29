
import os
import prody as pr
import sidechainnet as scn
from sidechainnet.utils.measure import get_seq_coords_and_angles
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import pickle
import argparse

from Bio import pairwise2

from tqdm.contrib.concurrent import process_map


def process_complex(path_tuple):
    complex_path, complex = path_tuple

    chains = []
    destination = os.path.join(complex_path, 'scn_complex.pkl')

    # ==============================
    # small hack
    # if not os.path.exists(destination):
    #     print(f'{destination} is non existent')
    #     return
    #
    # with open(destination, 'rb') as file:
    #     obj = pickle.load(file)
    #
    # print(obj['chn'].shape)
    # assert len(obj['chn'].shape) == 1
    # ==============================


    # if os.path.exists(destination):
    #     print(f'{destination} already converted!')
    #     return

    for i, chain in enumerate(('chain_1.pdb', 'chain_2.pdb', 'chain_1_unbound.pdb', 'chain_2_unbound.pdb')):
        chain = os.path.join(complex_path, chain)
        if not os.path.exists(chain): continue

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
            "crd": coords.reshape(-1, 14, 3),
            "seq": sequence,
            "chn": np.full([len(sequence)], i % 2, dtype=int)
        }
        chains.append(data)

    if len(chains) == 2:
        subunit1, subunit2 = chains
        data = {
            'ids': complex,
            'ang': np.concatenate((subunit1['ang'], subunit2['ang']), axis=0),
            'crd': np.concatenate((subunit1['crd'], subunit2['crd']), axis=0),
            'seq': subunit1['seq'] + subunit2['seq'],
            'chn': np.concatenate((subunit1['chn'], subunit2['chn']), axis=0)
        }
    else:
        bound1, bound2, unbound1, unbound2 = chains

        data = []
        for idx, (bound, unbound) in enumerate([(bound1, unbound1),
                                                (bound2, unbound2)]):
            records  = [''.join(list(data['seq'])) for data in (bound, unbound)]
            aligned = pairwise2.align.globalxx(*records, one_alignment_only=True)[0]

            length = len(aligned[0])

            bound_seq = np.array([char != '-' for char in aligned.seqA])
            unbound_seq = np.array([char != '-' for char in aligned.seqB])

            ang = np.zeros([length, 12])
            crd = np.zeros([length, 14, 3])
            tgt_crd = np.zeros([length, 14, 3])
            chn = np.full([length], idx)

            ang[unbound_seq] = unbound['ang']
            crd[unbound_seq] = unbound['crd']
            tgt_crd[bound_seq] = bound['crd']

            data.append(dict(ang=ang, tgt_crd=tgt_crd, crd=crd, chn=chn, seq=aligned.seqA))

        data = {
            'ids': complex,
            'ang': np.concatenate((data[0]['ang'], data[1]['ang']), axis=0),
            'crd': np.concatenate((data[0]['crd'], data[1]['crd']), axis=0),
            'tgt_crd': np.concatenate((data[0]['tgt_crd'], data[1]['tgt_crd']), axis=0),
            'seq': data[0]['seq'] + data[1]['seq'],
            'chn': np.concatenate((data[0]['chn'], data[1]['chn']), axis=0)
        }

    with open(destination, 'wb') as file:
        pickle.dump(data, file)

    print(f'{destination} just converted!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to embed sequences')

    parser.add_argument('--source', type=str, default='../../data/dips_preprocessed/')
    parser.add_argument('--split', type=str, default='DB5')
    parser.add_argument('--max_workers', type=int, default=40)

    args = parser.parse_args()
    source = os.path.join(args.source, args.split)

    paths = []
    for section in os.listdir(source):
        section_path = os.path.join(source, section)
        for complex in os.listdir(section_path):
            paths.append((os.path.join(section_path, complex), complex))
    paths = paths[::-1]

    if args.max_workers > 0:
        process_map(process_complex, paths, max_workers=args.max_workers)
    else:
        for path in tqdm(paths):
            process_complex(path)
