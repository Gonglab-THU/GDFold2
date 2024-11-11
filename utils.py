import os
import json
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from biopandas.pdb import PandasPdb
from pyrosetta import *

int2AA = {0:'GLY', 1:'ALA', 2:'CYS', 3:'GLU', 4:'ASP', 5:'PHE', 6:'ILE', 7:'HIS', 8:'LYS', 
          9:'MET', 10:'LEU', 11:'ASN', 12:'GLN', 13:'PRO', 14:'SER', 15:'ARG', 16:'THR', 
          17:'TRP', 18:'VAL', 19:'TYR'}
aa2int = {'G':0, 'A':1, 'C':2, 'E':3, 'D':4, 'F':5, 'I':6, 'H':7, 'K':8, 'M':9, 'L':10, 
          'N':11, 'Q':12, 'P':13, 'S':14, 'R':15, 'T':16, 'W':17, 'V':18, 'Y':19}
symbols = ['N', 'CA', 'C', 'O', 'CB', 'CEN', 'H']
elements = ['N', 'C', 'C', 'O', 'C', 'X' ,'H']

def load_fasta(fasta):
    sequences = SeqIO.parse(fasta, "fasta")
    for sequence in sequences:
        name = sequence.id
        seq = [aa2int[s] for s in sequence.seq]
    return name, seq

def all2cen(input, output):
    init("-mute all")
    pose = pose_from_file(input)
    switch_cen = SwitchResidueTypeSetMover("centroid")
    switch_cen.apply(pose)
    pose.dump_pdb(output)

def load_pred(pred, mode='SPIRED'):
    data = np.load(pred)
    trans = lambda x: torch.FloatTensor(x)
    if mode == 'SPIRED':
        info = {'reference': list(data['reference']), 
                'translation': trans(data['translation']),
                'dihedrals': trans(data['dihedrals']),
                'plddt': trans(data['plddt'])}
    if mode == 'Cerebra':
        info = {'reference': list(data['reference']), 
                'rotation': trans(data['rotation']),
                'translation': trans(data['translation']), 
                'dihedrals': trans(data['dihedrals']),
                'plddt': trans(data['plddt'])}
    if mode == 'Dynamics':
        info = {'reference': trans(data['reference']), 
                'rotation': trans(data['rotation']),
                'translation': trans(data['translation'])}
    if mode == 'Rosetta':
        info = {'dist': trans(data['dist']).permute(2, 0, 1),
                'omega': trans(data['omega']).permute(2, 0, 1),
                'theta': trans(data['theta']).permute(2, 0, 1),
                'phi': trans(data['phi']).permute(2, 0, 1)}
    return info

def get_params(seq):
    params = {}
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(scriptdir, 'params.json')) as jsonfile:
        statistic = json.load(jsonfile)
    for key in statistic.keys():
        params[key] = torch.FloatTensor(np.array(statistic[key]))
    params['vdw_mask'] = torch.IntTensor(vdw_mask(seq))
    params['vdw_dist'] = vdw_dist(seq, params['vdw_radius'])
    return params

def vdw_mask(seq):
    length = len(seq) * 7
    mask = np.ones((length, length))
    mask -= np.diag(mask.diagonal())
    gly_index = [i for i, x in enumerate(seq) if x == 0]
    pro_index = [i for i, x in enumerate(seq) if x == 13]
    for i in range(length):
        if i % 7 == 0: mask[i:i + 5, i:i + 5] = 0
        if (i - 1) % 7 == 0 and (i + 8) < length: mask[i, i + 8] = 0
        if (i - 1) % 7 == 0: mask[i, i + 4] = 0
        if (i - 2) % 7 == 0: mask[i, i + 4] = 0
    for gly in gly_index:
        mask[gly * 7 + 3] = 0
        mask[:, gly * 7 + 3] = 0
    for pro in pro_index:
        mask[pro * 7 + 6] = 0
        mask[:, pro * 7 + 6] = 0
    mask = np.triu(mask)
    mask += mask.T
    return mask

def vdw_dist(seq, vdw_radius):
    vdw_radius = vdw_radius.repeat(len(seq))
    vdw_dist = vdw_radius.unsqueeze(0) + vdw_radius.unsqueeze(1) - 1.2
    return vdw_dist

def dump2pdb(path, name, seq, coords):
    sequence = [int2AA.get(s) for s in seq]
    CA, C, N, CB, CEN, O, H = coords
    x_coord, y_coord, z_coord = [], [], []
    for l in range(len(sequence)):
        for atom_type in [N, CA, C, O, CB, CEN, H]:
            x_coord.append(float('{:.3f}'.format(atom_type[l][0])))
            y_coord.append(float('{:.3f}'.format(atom_type[l][1])))
            z_coord.append(float('{:.3f}'.format(atom_type[l][2])))
    length = 7 * len(sequence)
    atoms = pd.DataFrame({"record_name": ["ATOM"] * length, "atom_number": list(np.arange(1, length+1)), "blank_1": [''] * length,
                          "atom_name": ['N', 'CA', 'C', 'O', 'CB', 'CEN', 'H'] * len(sequence), "alt_loc": [''] * length,
                          "residue_name": [res for res in sequence for _ in range(7)], "blank_2": [''] * length, "chain_id": ["A"]  * length, 
                          "residue_number": [n+1 for n in range(len(sequence)) for _ in range(7)], "insertion": [''] * length, "blank_3": [''] * length, 
                          "x_coord": x_coord, "y_coord": y_coord, "z_coord": z_coord, "occupancy": [1.0] * length, "b_factor": [0.0] * length, 
                          "blank_4": [''] * length, "segment_id": [''] * length, "element_symbol": ['N', 'C', 'C', 'O', 'C', 'X' ,'H'] * len(sequence), 
                          "charge": [np.nan] * length, "line_idx": list(np.arange(length))})
    atoms.drop(atoms[(atoms.residue_name == "GLY") & (atoms.atom_name == "CB")].index, inplace=True)
    atoms.drop(atoms[(atoms.residue_name == "PRO") & (atoms.atom_name == "H")].index, inplace=True)
    atoms["atom_number"] = list(np.arange(1, len(atoms)+1))
    atoms["line_idx"] = list(np.arange(len(atoms)))
    pdb_parse = PandasPdb()
    pdb_parse.df['ATOM'] = atoms
    pdb_parse.to_pdb(os.path.join(path, name))
