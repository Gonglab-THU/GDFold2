import sys, os
import torch
import argparse
import numpy as np
from Bio import PDB
from biopandas.pdb import PandasPdb

parser = argparse.ArgumentParser(description="Extract geometric constraints from PDB files")
parser.add_argument('--state1', type=str, help='conformational state 1')
parser.add_argument('--state2', type=str, help='conformational state 2')
parser.add_argument('--output_path', type=str, help='output directory path')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

def pdb2fasta(pdb_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure(id=None, file=pdb_file)
    assert len(structure) == 1, "Input structure must have only one model!"
    model = structure[0]

    chain_ids = []
    for chain in model:
        chain_ids.append(chain.id)
    assert len(chain_ids) == 1, "Input model must have only one chain!"
    chain_id = chain_ids[0]

    sequence = []
    for residue in model[chain_id]:
        if PDB.is_aa(residue):
            sequence.append(PDB.Polypeptide.three_to_one(residue.resname))
        else:
            print(f"Non-standard residue ({residue}) is not allowed in the sequence!")
            sys.exit(0)
    sequence = ''.join(sequence)
    return sequence

class Protein:
    def __init__(self, pdb_file):
        self.coords = self.get_coords(pdb_file)
        self.frames = self.build_frames(self.coords)
        self.translation = self.transform_translation(self.coords, self.frames)
        self.rotation = self.transform_rotation(self.frames)

    def get_coords(self, pdb_file):
        pdb_data = PandasPdb().read_pdb(pdb_file)
        atoms = pdb_data.df['ATOM']
        [CA, C, N] = [[] for _ in range(3)]
        for i in range(len(atoms)):
            x, y, z = atoms.loc[i, 'x_coord'], atoms.loc[i, 'y_coord'], atoms.loc[i, 'z_coord']
            if atoms.loc[i, 'atom_name'] == 'CA': CA.append([x, y, z])
            if atoms.loc[i, 'atom_name'] == 'C': C.append([x, y, z])
            if atoms.loc[i, 'atom_name'] == 'N': N.append([x, y, z])
        assert len(CA) == len(C) == len(N), "Heavy-atoms in main-chain are missing!"
        coords = torch.stack([torch.FloatTensor(atom) for atom in [N, CA, C]], dim=1)
        return coords
    
    def build_frames(self, coords):
        frames = []
        for res in coords:
            x1 = res[0]
            x2 = res[1]
            x3 = res[2]
            origin = x2
            v1 = x3 - x2
            v2 = x1 - x2
            e1 = v1 / torch.norm(v1)
            n2 = v2 - torch.dot(v2, e1) * e1
            e2 = n2 / torch.norm(n2)
            e3 = torch.cross(e1,e2)
            frames.append(torch.stack((origin, e1, e2, e3)))
        return torch.stack(frames)
    
    def transform_translation(self, coords, frames):
        origins = frames[:, 0, :]
        R = frames[:, 1:4, :]
        ca_coords = coords[:, 1, :]
        translated = ca_coords.unsqueeze(0) - origins.unsqueeze(1)
        N = translated.permute(0,2,1)
        translation = torch.einsum('iab,ibc->iac', R, N).permute(0,2,1)
        return translation
    
    def transform_rotation(self, frames):
        R = frames[:, 1:4, :].permute(0, 2, 1)
        R_transposed = R.permute(0, 2, 1)
        rotation = torch.matmul(R_transposed.unsqueeze(1), R.unsqueeze(0))
        return rotation

def parse_pdbs(state1, state2):
    fasta1 = pdb2fasta(state1)
    fasta2 = pdb2fasta(state2)
    assert fasta1 == fasta2, "Input conformations must have the same protein sequences!"

    with open(os.path.join(args.output_path, "comb.fasta"), 'w') as f:
        f.write(">comb\n")
        f.write(fasta1)

    length = len(fasta1)
    protein1 = Protein(state1)
    protein2 = Protein(state2)

    n_reference = min(32, length)
    reference = torch.randperm(length)[:n_reference]

    translations = torch.stack((protein1.translation[reference], protein2.translation[reference]), dim=0).numpy()
    rotations = torch.stack((protein1.rotation[reference], protein2.rotation[reference]), dim=0).numpy()
    reference = reference.unsqueeze(0).repeat(2, 1).numpy()
    return reference, translations, rotations

def main():
    reference, translations, rotations = parse_pdbs(args.state1, args.state2)
    try:
        file_name = os.path.join(args.output_path, "comb.npz")
        np.savez(file_name, reference = reference, translation = translations, rotation = rotations)
        print(f"Converting done! Fasta file (comb.fasta) and constraints file (comb.npz) are saved in {args.output_path}")
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()