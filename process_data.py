import os
import torch
import math
import shutil
import csv
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import PDB
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from torch_geometric.data import Data
from rdkit import Chem
import networkx as nx
from rdkit.Chem import BondType
import warnings
warnings.filterwarnings("ignore")

# aa type
amino_acid_map = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
    'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}
id_vab = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10,
          'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20,
          'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30,
          'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35}
# d map
drugSeq_vocab = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
			      ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
			      "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
			      "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			      "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
			      "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
			      "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
			      "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
			      "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
			      "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			      "t": 61, "y": 62, "@": 63, "/": 64, ":": 65, "~": 66}

def get_id_encoder(pdb_id):
    id_tensor = []
    for i in list(pdb_id):
        if i in id_vab.keys():
            id_tensor.append(id_vab[i])
        else:
            print(pdb_id)
    return torch.tensor(id_tensor).type(torch.int64)

def get_Ca_coord(residues):
    Ca_coord = []
    for res in residues:
        try:
            coord = res.child_dict['CA'].coord
        except:
            print(res)
        Ca_coord.append(coord)
    return Ca_coord

def get_sequence(structure):
    residues_list = [residue for residue in structure.get_residues() if residue.get_id()[0] == ' ']
    residues = []
    for res in residues_list:
        if 'CA' in res.child_dict.keys():
            residues.append(res)
    seqs = ''.join([seq1(residue.get_resname()) for residue in residues])
    Ca_coord = get_Ca_coord(residues)
    return residues, seqs, Ca_coord

def seq_cat(prot):
    x = np.zeros(len(prot))
    for i, ch in enumerate(prot):
        if ch in amino_acid_map:
            x[i] = amino_acid_map[ch]
        else:
            x[i] = max(amino_acid_map.values()) + 1
            amino_acid_map[ch] = x[i]
    return torch.from_numpy(x).type(torch.int64)

def generate_mask(seq_res, sequence, pocket_res):
    mask = len(sequence) * [0]
    pock_f_id = [res_i.full_id for res_i in pocket_res]
    prot_f_id = [res_j.full_id for res_j in seq_res]
    for res_i in pocket_res:
        for res_j in seq_res:
            if res_i.full_id == res_j.full_id:
                index = seq_res.index(res_j)
                mask[index] = 1

    idx_1_list = []
    for i, ele in enumerate(mask):
        if ele == 1:
            idx_1_list.append(i)
    for id_i in pock_f_id:
        p_f_id = [prot_f_id[id_j] for id_j in idx_1_list]
        if id_i in p_f_id:
            new_mask = mask
        else:
            print(sequence, 'error')
    return mask

def amino_acid_to_one_hot(amino_acid):
    one_hot_encoded = np.zeros(len(amino_acid_map))
    one_hot_encoded[amino_acid_map[amino_acid]-1] = 1
    return one_hot_encoded

def protein_to_high_dimensional(protein_sequence):
    one_hot_encoded = [amino_acid_to_one_hot(amino_acid) for amino_acid in protein_sequence]
    high_dimensional_vector = torch.LongTensor(np.array(one_hot_encoded))
    return high_dimensional_vector

def calculate_distance(ca_i, ca_j):
    squared_distance = sum([(x - y) ** 2 for x, y in zip(ca_i, ca_j)])
    distance = math.sqrt(squared_distance)
    return distance

def get_graph(pocket_seq, pocket_Ca_pos, pocket_res_dis):
    node = protein_to_high_dimensional(pocket_seq)
    edge = np.zeros((len(pocket_seq), len(pocket_seq)))
    for i in range(len(pocket_seq)):
        for j in range(len(pocket_seq)):
            distance = calculate_distance(pocket_Ca_pos[i], pocket_Ca_pos[j])
            if distance <= pocket_res_dis:
                edge[i][j] = 1
            else:
                edge[i][j] = 0
    edge_ = torch.tensor(np.argwhere(edge == 1).T)
    return node, edge_

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
               for x in (Chem.rdchem.BondType.SINGLE, \
                         Chem.rdchem.BondType.DOUBLE, \
                         Chem.rdchem.BondType.TRIPLE, \
                         Chem.rdchem.BondType.AROMATIC,\
                         Chem.rdchem.BondType.IONIC,\
                         Chem.rdchem.BondType.DATIVE,\
                         Chem.rdchem.BondType.HYDROGEN,\
                         Chem.rdchem.BondType.THREECENTER,\
                         Chem.rdchem.BondType.DATIVEL,\
                         Chem.rdchem.BondType.DATIVER)]
        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t
    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))
    return edge_index, edge_attr
def atom_features(mol, graph,
                  atom_symbols=['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
                                'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                                'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']):
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                  one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
                  one_of_k_encoding_unk(atom.GetNoImplicit, [0, 1, 2, 3, 4, 5]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) + \
                  [atom.GetIsAromatic()]
        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))
def mol2graph(mol):
    if mol is None:
        return None
    g = nx.DiGraph()
    atom_features(mol, g)
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                           b_type=e_ij.GetBondType(),
                           IsConjugated=int(e_ij.GetIsConjugated()),
                           )
    x = torch.stack([feats['feats'] for n, feats in g.nodes(data=True)])
    edge_index, edge_attr = get_edges(g)
    return x, edge_index, edge_attr

def smiles_to_tensor(smiles, token_dict):
    smiles_tokens = []
    for char in smiles:
        smiles_tokens.append(char)
    drug_seqs = []
    for token in smiles_tokens:
        if token in token_dict:
            label = token_dict[token]
        else:
            label = max(token_dict.values()) + 1
            token_dict[token] = label
        drug_seqs.append(label)
    smiles_seq = torch.tensor(drug_seqs).type(torch.int64)
    return smiles_seq

def ligand_process(labels_dict, complex_files, pdb_id):
    aff = labels_dict[pdb_id]
    y = torch.FloatTensor([aff])

    ligand_path_mol2 = os.path.join(complex_files, pdb_id + '_ligand.mol2')
    ligand_path_sdf = os.path.join(complex_files, pdb_id + '_ligand.sdf')
    mols = Chem.MolFromMol2File(ligand_path_mol2)
    if mols is None:
        mols = Chem.SDMolSupplier(ligand_path_sdf)[0]
    if mols is None:
        mols = Chem.MolFromMolFile(ligand_path_sdf, sanitize=False)
    try:
        x, edge_index, edge_attr = mol2graph(mols)
        smiles = Chem.MolToSmiles(mols)
        smiles_seq = smiles_to_tensor(smiles, drugSeq_vocab)
    except:
        print(f'{pdb_id} ligand process error')
    ligand_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    smiles_data = Data(x=smiles_seq)

    return ligand_graph, smiles_data

def generate_fea(args):
    datasets_path = args.data_path
    save_path = args.save_path
    labels_path = args.labels_path
    pocket_size = args.pocket_size
    pocket_res_dis = args.pocket_res_dis
    max_length = args.max_length

    labels = pd.read_csv(labels_path)
    labels_dict = {item[0]: item[1] for item in labels.values}

    for id in tqdm(labels_dict.keys(), ncols=80):
        complex_files = os.path.join(datasets_path, id)
        id_dict = {}
        pdb_id = complex_files.split('/')[-1]
        id_tesnor = get_id_encoder(pdb_id)
        id_dict[pdb_id] = id_tesnor

        if not os.path.exists(save_path + f'/{pdb_id}_feature.pt'):
            # protein
            prot_path = os.path.join(complex_files, pdb_id + '_protein.pdb')
            pock_path = os.path.join(complex_files, f'Pocket_{pocket_size}A.pdb')
            prot_parser = PDBParser()
            protein = prot_parser.get_structure(pdb_id, prot_path)
            pocket = prot_parser.get_structure(pdb_id, pock_path)
            prot_res, prot_seq, _ = get_sequence(protein)
            pocket_res, pocket_seq, pocket_Ca_pos = get_sequence(pocket)

            prot_encoder = seq_cat(prot_seq)
            mask = generate_mask(prot_res, prot_seq, pocket_res)

            pock_node, pock_edge = get_graph(pocket_seq, pocket_Ca_pos, pocket_res_dis)
            # val_num
            if mask.count(1) == pock_node.size(0):
                p_seq = Data(x=prot_encoder, y=torch.tensor(mask), edge_attr=id_tesnor)
                pocket_graph = Data(x=pock_node, edge_index=pock_edge)
            else:
                print(pdb_id, 'pocket residue num not match')

            # ligand
            ligand_graph, smiles_data = ligand_process(labels_dict, complex_files, pdb_id)

            fea_dict = {'seq_prot': p_seq, 'graph_pocket': pocket_graph, 'graph_ligand': ligand_graph, 'smiles_ligand': smiles_data}

            # torch.save(fea_dict, save_path + f'/{pdb_id}_feature.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # datasets parser
    parser.add_argument('--data_path', type=str, default='/media/ST-18T/ljr/Datasets/PDBbind_2020/19443')
    parser.add_argument('--labels_path', type=str, default='/media/ST-18T/ljr/Datasets/MHAI/18809.csv')
    parser.add_argument('--save_path', type=str, default='/media/ST-18T/ljr/Datasets/MHAI/18809')
    # fea parser
    parser.add_argument('--pocket_size', type=int, default=5, help='pocket size') # 4 5 6
    parser.add_argument('--pocket_res_dis', type=int, default=6, help='pocket residue distance') # 4 5 6
    parser.add_argument('--max_length', type=int, default=1400, help='seq max length') # 1300 1400 1500

    args = parser.parse_args()
    generate_fea(args)



