import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
from scipy.spatial.transform import Rotation as R
from biopandas.pdb import PandasPdb

int2AA = {0:'GLY', 1:'ALA', 2:'CYS', 3:'GLU', 4:'ASP', 5:'PHE', 6:'ILE', 7:'HIS', 
          8:'LYS', 9:'MET', 10:'LEU', 11:'ASN', 12:'GLN', 13:'PRO', 14:'SER', 
          15:'ARG', 16:'THR', 17:'TRP', 18:'VAL', 19:'TYR'}
AA2int = {k:v for v,k in int2AA.items()}

def load_pdb(pdb):
    pdb_data = PandasPdb().read_pdb(pdb)
    atoms = pdb_data.df['ATOM']
    sequence = []
    for i in range(len(atoms)):
        if atoms.loc[i, 'atom_name'] == 'CA':
            sequence.append(atoms.loc[i, 'residue_name'])
    seq = [AA2int[res] for res in sequence]
    [CA, C, N, CB, CEN, O, H] = [[] for _ in range(7)]
    for i in range(len(atoms)):
        x, y, z = atoms.loc[i, 'x_coord'], atoms.loc[i, 'y_coord'], atoms.loc[i, 'z_coord']
        if atoms.loc[i, 'atom_name'] == 'CA': CA.append([x, y, z])
        if atoms.loc[i, 'atom_name'] == 'C': C.append([x, y, z])
        if atoms.loc[i, 'atom_name'] == 'N': 
            N.append([x, y, z])
            if atoms.loc[i, 'residue_name'] == 'PRO':
                H.append([x, y, z])
        if atoms.loc[i, 'atom_name'] == 'CB': CB.append([x, y, z])
        if atoms.loc[i, 'atom_name'] == 'CEN': 
            CEN.append([x, y, z])
            if atoms.loc[i, 'residue_name'] == 'GLY':
                CB.append([x, y, z])
        if atoms.loc[i, 'atom_name'] == 'O': O.append([x, y, z])
        if atoms.loc[i, 'atom_name'] == 'H': H.append([x, y, z])
    coords = np.array([CA, C, N, CB, CEN, O, H])
    return seq, coords

class GraphData:
    def __init__(self, coords, seq, label, n_embed=16, 
                 top_k=30, n_rbf=16, cutoff=8.):
        self.coords = coords
        self.seq = seq
        self.label = label
        self.n_embed = n_embed
        self.top_k = top_k
        self.n_rbf = n_rbf
        self.cutoff = cutoff

    def features(self):
        with torch.no_grad():
            coords = torch.FloatTensor(self.coords)
            seq = torch.IntTensor(self.seq)
            label = torch.FloatTensor(self.label)
            
            # info
            CA_coord = coords[0]
            edge_index = torch_cluster.knn_graph(CA_coord, k=self.top_k)
            edge_vectors = CA_coord[edge_index[0]] - CA_coord[edge_index[1]]

            # node scalar features: [n_nodes, 12]
            dihedrals = self._dihedrals(coords)
            angles = self._angles(coords, seq)
            bonds = self._bonds(coords)
            node_s = torch.cat([dihedrals, angles, bonds], dim=-1)

            # node vector features: [n_nodes, 3, 3]
            orientation = self._orientations(CA_coord)
            sidechains = self._sidechains(coords)
            node_v = torch.cat([orientation, sidechains.unsqueeze(-2)], dim=-2)

            # edge scalar features: [n_edges, 38]
            pos_embedding = self._pos_embedding(edge_index)
            rbf = self._rbf(torch.linalg.norm(edge_vectors, dim=-1))
            hbond = self._hbond(coords, edge_index)
            rotation = self._rotation(coords, edge_index)
            contact = self._contact(coords[3], edge_index)
            edge_s = torch.cat([pos_embedding, rbf, hbond, rotation, contact], dim=-1)

            # edge vector features: [n_edges, 1, 3]
            edge_v = F.normalize(edge_vectors).unsqueeze(-2)

        data = {'seq': seq, 'label': label, 
                'node_s': node_s, 'node_v': node_v,
                'edge_s': edge_s, 'edge_v': edge_v, 
                'edge_index': edge_index}
        return data
    
    def _dihedrals(self, coords, eps=1e-8):
        '''
        From https://github.com/jingraham/neurips19-graph-protein-design
        '''
        # dihedrals features, [n_nodes, 6]
        X = torch.stack([coords[2], coords[0], coords[1]], dim=-2).view(-1, 3)

        # shifted slices of unit vectors
        dX = X[1:] - X[:-1]
        U = F.normalize(dX, dim=-1)
        u_0 = U[2:]
        u_1 = U[1:-1]
        u_2 = U[:-2]

        # backbone normals
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)

        # angle between normals
        cosD = torch.sum(n_2 * n_1, dim=-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign(torch.sum(u_2 * n_1, dim=-1)) * torch.arccos(cosD)

        # remove phi[0], psi[-1] and omega[-1]
        D = F.pad(D, [1, 2]).view(-1, 3)
        D_features = torch.cat((torch.cos(D), torch.sin(D)), dim=1)
        return D_features
    
    def _angles(self, coords, seq):
        # angle features, [n_nodes, 4]
        o_vec = F.normalize(coords[-2, :-1] - coords[1, :-1])
        v_1 = F.normalize(coords[0, :-1] - coords[1, :-1])
        v_2 = F.normalize(coords[2, 1:] - coords[1, :-1])
        
        angle_1 = F.pad(torch.arccos(torch.sum(v_1 * o_vec, dim=-1)), [0, 1])
        angle_2 = F.pad(torch.arccos(torch.sum(v_2 * o_vec, dim=-1)), [0, 1])
        o_feature = torch.stack([angle_1, angle_2], dim=-1)
        
        h_vec = F.normalize(coords[-1, 1:] - coords[2, 1:])
        v_3 = F.normalize(coords[0, 1:] - coords[2, 1:])
        v_4 = F.normalize(coords[1, :-1] - coords[2, 1:])
        
        pro_mask = torch.ones(len(seq))
        pro_index = [i for i, x in enumerate(seq) if x == 13]
        pro_mask[pro_index] = 0.0
        angle_3 = F.pad(torch.arccos(torch.sum(v_3 * h_vec, dim=-1)), [1, 0])
        angle_4 = F.pad(torch.arccos(torch.sum(v_4 * h_vec, dim=-1)), [1, 0])
        h_feature = pro_mask[:, None] * torch.stack([angle_3, angle_4], dim=-1)

        A_feature = torch.sin(torch.cat([o_feature, h_feature], dim=1))
        return A_feature
    
    def _bonds(self, coords):
        # peptide bond features, [n_nodes, 2]
        peptide_bond = torch.linalg.norm(coords[2, 1:] - coords[1, :-1], dim=-1)
        peptide_bond = torch.clamp(abs(peptide_bond - 1.33) - 0.01, min=0.)
        # remove backward[0], forward[-1]
        bond_forward = F.pad(peptide_bond, [0, 1])
        bond_backward = F.pad(peptide_bond, [1, 0])
        bond_feature = torch.stack([bond_forward, bond_backward], dim=-1)
        return bond_feature
    
    def _orientations(self, coord):
        # orientations of CA atoms, [n_nodes, 2, 3]
        forward = F.pad(F.normalize(coord[1:] - coord[:-1]), [0, 0, 0, 1])
        backward = F.pad(F.normalize(coord[1:] - coord[:-1]), [0, 0, 1, 0])
        orien_feature = torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)
        return orien_feature
    
    def _sidechains(self, coords):
        # vector of sidechains (CA -> CEN), [n_nodes, 1, 3]
        vec = F.normalize(coords[4] - coords[0])
        return vec

    def _pos_embedding(self, edge_index):
        '''
        From https://github.com/jingraham/neurips19-graph-protein-design
        '''
        # position embedding, [n_edges, 16]
        position = edge_index[0] - edge_index[1]
        frequency = torch.exp(
            torch.arange(0, self.n_embed, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.n_embed))
        angles = position.unsqueeze(-1) * frequency
        embeddings = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return embeddings
    
    def _rbf(self, D, D_min=0., D_max=20., D_count=16):
        '''
        From https://github.com/jingraham/neurips19-graph-protein-design
        '''
        # distance radial basis function, [n_edges, 16]
        D_mu = torch.linspace(D_min, D_max, D_count)
        D_mu = D_mu.view(1, -1)
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        rbf = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return rbf

    def _hbond(self, coords, edge_index):
        '''
        From https://github.com/jingraham/neurips19-graph-protein-design
        '''
        # hydrogen bond features, [n_edges, 1]
        def _inv_distance(atom_1, atom_2):
            distance = torch.linalg.norm(atom_1 - atom_2, dim=-1)
            return 1. / (distance + 1e-8)
        # acceptor (O, C)
        accept_coords = (coords[-2].unsqueeze(1), coords[1].unsqueeze(1))
        # donor (N, H)
        donor_coords = (coords[2][edge_index[0]].view(-1, self.top_k, 3),
                        coords[-1][edge_index[0]].view(-1, self.top_k, 3))
        # vacuum electrostatics
        U = (0.084 * 332) * (_inv_distance(accept_coords[0], donor_coords[0])
                            + _inv_distance(accept_coords[1], donor_coords[1])
                            - _inv_distance(accept_coords[0], donor_coords[1])
                            - _inv_distance(accept_coords[1], donor_coords[0]))
        neighbor_mask = 1 - (abs(edge_index[0] - 
                                 edge_index[1]) == 1).type(torch.float32).view(-1, self.top_k)
        HB = (neighbor_mask * (U < -0.5).type(torch.float32)).view(-1, 1)
        return HB
    
    def _contact(self, coord, edge_index):
        '''
        From https://github.com/jingraham/neurips19-graph-protein-design
        '''
        # CB-CB contacts, [n_edges, 1]
        dist = torch.linalg.norm(coord[edge_index[0]] - 
                                 coord[edge_index[1]], dim=-1)
        contact = (dist < self.cutoff).type(torch.float32).view(-1, 1)
        return contact

    def _rotation(self, coords, edge_index):
        # rotation features: [n_edges, 4]
        x_axis = F.normalize(coords[1] - coords[0])
        z_axis = F.normalize(torch.cross(x_axis, coords[2] - coords[0]))
        y_axis = torch.cross(z_axis, x_axis)
        axis = torch.cat([x_axis.unsqueeze(1), 
                          y_axis.unsqueeze(1), 
                          z_axis.unsqueeze(1)], dim=-2)
        U, S, VH = torch.linalg.svd(axis)
        # rotation matrix in global coordinate system
        rot_mat = torch.inverse(VH) @ torch.inverse(U)
        # relative rotation matrix to the reference residue
        relative_rot = torch.inverse(rot_mat).unsqueeze(1) @ rot_mat[edge_index[0]].view(-1, self.top_k, 3, 3)
        rot_quat = torch.FloatTensor(R.from_matrix(relative_rot.view(-1, 3, 3)).as_quat())
        return rot_quat