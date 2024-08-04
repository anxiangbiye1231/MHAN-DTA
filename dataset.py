import pickle
import copy
import numpy
from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import multiprocessing

def slice_up(seqs_trans_list, prot_len):
    seq_prot_list = []
    for seq in seqs_trans_list:
        seq_prot = seq.x.numpy()
        seq_prot = seq_prot[:prot_len].tolist() + [0] * max(0, prot_len - seq_prot.size)
        seq_prot = Data(x=torch.tensor(seq_prot).unsqueeze(0))
        seq_prot_list.append(seq_prot)
    return seq_prot_list

class Diadataset(Dataset):
    def __init__(self, pdb_files, labels, prot_len, pock_dis):
        self.pdb_files = pdb_files
        self.labels = labels
        self.prot_len = prot_len
        self.pock_dis = pock_dis
        self.pre_process()

    def pre_process(self):
        data_dir = self.pdb_files
        labels = self.labels
        prot_len = self.prot_len
        pock_dis = self.pock_dis
        data_list = []
        seq_trans = []
        graph_pocket = []
        smiles_ligand = []
        graph_ligand = []

        for pdb_id in tqdm(labels.keys(), ncols=80):
            if os.path.exists(data_dir + f'/{pdb_id}_feature.pt'):
                with open(data_dir + f'/{pdb_id}_feature.pt', 'rb') as f:
                    pdb_data = torch.load(f)
                data_list.append(pdb_data)
                seq_trans.append(pdb_data['seq_prot'])
                graph_pocket.append(pdb_data['graph_pocket'])
                smiles_ligand.append(pdb_data['smiles_ligand'])
                graph_ligand.append(pdb_data['graph_ligand'])
        seqs_prot = slice_up(seq_trans, prot_len)

        self.seq_trans = seq_trans
        self.pocket_res_graph = graph_pocket
        self.seq_info = seqs_prot
        self.ligand_graph = graph_ligand
        self.ligand_smiles = smiles_ligand
        self.datasets_num = len(seq_trans)

    def __getitem__(self, index):
        seq_trans = self.seq_trans[index]
        pocket_res_graph = self.pocket_res_graph[index]
        seq_info = self.seq_info[index]
        ligand_graph = self.ligand_graph[index]
        ligand_smiles = self.ligand_smiles[index]
        return seq_trans, pocket_res_graph, seq_info, ligand_graph, ligand_smiles

    def __len__(self):
        return self.datasets_num

def seq_padding(data_list):
    prot_seq = [idx[0] for idx in data_list]
    # seq
    p_seq_x = [idx.x for idx in prot_seq]
    max_len = max([id.shape[0] for id in p_seq_x])
    zero_numpy_x = numpy.zeros((len(data_list), max_len))
    for idx, seq in enumerate(p_seq_x):
        seq_x = seq.numpy()
        try:
            zero_numpy_x[idx, :len(seq_x)] = seq_x
        except:
            print(idx,max_len, len(seq_x), zero_numpy_x.shape, prot_seq)
            pass
    # seq_mask
    p_seq_y = [idx.y for idx in prot_seq]
    max_len = max([id.shape[0] for id in p_seq_y])
    zero_numpy_y = numpy.zeros((len(data_list), max_len))
    for idx, seq in enumerate(p_seq_y):
        seq_y = seq.numpy()
        zero_numpy_y[idx, :len(seq_y)] = seq_y
    return zero_numpy_x, zero_numpy_y

def smiles_padding(data_list):
    ligand_seq = [idx[4] for idx in data_list]
    l_seq = [idx.x for idx in ligand_seq]
    max_len = max([id.shape[0] for id in l_seq])
    zero_numpy_l = numpy.zeros((len(data_list), max_len))
    for idx, seq in enumerate(l_seq):
        seq_x = seq.numpy()
        zero_numpy_l[idx, :len(seq_x)] = seq_x
    return zero_numpy_l

def collate_fn(data_list):
    new_data_list = copy.deepcopy(data_list)
    zero_numpy_x, zero_numpy_y = seq_padding(new_data_list)
    for id, index in enumerate(new_data_list):
        index[0].x = torch.from_numpy(zero_numpy_x[id]).to(torch.int64).unsqueeze(0)
        index[0].y = torch.from_numpy(zero_numpy_y[id]).to(torch.int64).unsqueeze(0)

    batch_seq_trans = Batch.from_data_list([data[0] for data in new_data_list])
    batch_res_graph = Batch.from_data_list([data[1] for data in new_data_list])
    batch_seq_info = Batch.from_data_list([data[2] for data in new_data_list])
    batch_lig_graph = Batch.from_data_list([data[3] for data in new_data_list])
    batch_lig_smiles = Batch.from_data_list([data[4] for data in new_data_list])
    return batch_seq_trans, batch_res_graph, batch_seq_info, batch_lig_graph, batch_lig_smiles