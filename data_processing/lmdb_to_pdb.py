""" modified from Atom3D
https://github.com/drorlab/atom3d
"""

import os
import pickle
import argparse

from collections import defaultdict
from functools import partial

import numpy as np
import torch
import os

import atom3d.datasets as da

from tqdm import tqdm

from atom3d.util.formats import df_to_bp
from Bio.PDB import PDBIO, Select

from tqdm.contrib.concurrent import process_map


triple_to_single_map = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V"
}
triple_to_single = defaultdict(lambda: "U")
triple_to_single.update(triple_to_single_map)

index_columns = \
    ['ensemble', 'subunit', 'structure', 'model', 'chain', 'residue']


class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0


def get_subunits(ensemble):
    # modified from atom3d
    subunits = ensemble['subunit'].unique()
    if len(subunits) == 4:
        lb = [x for x in subunits if x.endswith('ligand_bound')][0]
        lu = [x for x in subunits if x.endswith('ligand_unbound')][0]
        rb = [x for x in subunits if x.endswith('receptor_bound')][0]
        ru = [x for x in subunits if x.endswith('receptor_unbound')][0]
        bdf0 = ensemble[ensemble['subunit'] == lb]
        bdf1 = ensemble[ensemble['subunit'] == rb]
        udf0 = ensemble[ensemble['subunit'] == lu]
        udf1 = ensemble[ensemble['subunit'] == ru]
        names = (lb, rb, lu, ru)
    elif len(subunits) == 2:
        udf0, udf1 = None, None
        bdf0 = ensemble[ensemble['subunit'] == subunits[0]]
        bdf1 = ensemble[ensemble['subunit'] == subunits[1]]
        names = (subunits[0], subunits[1], None, None)
    else:
        raise RuntimeError('Incorrect number of subunits for pair')
    return names, (bdf0, bdf1, udf0, udf1)


def clean_backbone(df):
    df = df.reset_index(drop=True)
    chain = np.unique(df['chain'])[0]
    backbone = df[(df['name'] == 'CA') & (df['chain'] == chain)]
    return df, backbone['resname']
    

def lmdb_to_pdb(datum, destination, max_complex_size, min_complex_size, split):
    target_df = datum['atoms_pairs']
    sub_names, (bound1, bound2, unbound1, unbound2) = get_subunits(target_df)

    bound1, bound_chain1 = clean_backbone(bound1)
    bound2, bound_chain2 = clean_backbone(bound2)

    if (unbound1 is not None) and (unbound2 is not None):
        unbound1, unbound_chain1 = clean_backbone(unbound1)
        unbound2, unbound_chain2 = clean_backbone(unbound2)

    if not (max_complex_size > len(bound_chain1) + len(bound_chain2) > min_complex_size):
        return False

    ensemble = bound1.ensemble[0]

    for bound, filename in zip([bound1, bound2, unbound1, unbound2], ['chain_1.pdb', 'chain_2.pdb', 'chain_1_unbound.pdb', 'chain_2_unbound.pdb']):
        base_path = os.path.join(destination, split, ensemble[:2], ensemble)
        full_path = os.path.join(base_path, filename)

        if os.path.exists(full_path):
            print(f'{full_path} already converted!')
            continue

        struct = df_to_bp(bound)
        io = PDBIO()
        os.makedirs(base_path, exist_ok=True)
        io.set_structure(struct)
        io.save(full_path, NonHetSelect())

        print(f'{full_path} just converted!')

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to embed sequences')

    parser.add_argument('--destination', type=str, default='../../data/dips_preprocessed/')
    parser.add_argument('--split', type=str, default='DB5')
    parser.add_argument('--max_workers', type=int, default=0)

    parser.add_argument('--max_complex_size', type=int, default=5000)
    parser.add_argument('--min_complex_size', type=int, default=10)

    args = parser.parse_args()

    dataset = da.load_dataset(f'../../data/dips_raw/raw/DB5/data', 'lmdb')

    deconstructor = partial(
        lmdb_to_pdb,
        destination=args.destination,
        max_complex_size=args.max_complex_size,
        min_complex_size=args.min_complex_size,
        split=args.split
    )

    if args.max_workers == 0:
        ok = list()
        for datum in tqdm(dataset): ok.append(deconstructor(datum))
    else:
        ok = process_map(deconstructor, dataset, max_workers=args.max_workers)
        print('done')
