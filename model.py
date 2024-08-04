# %%
import torch
from torch import nn
from torch_geometric.nn import global_add_pool, GraphConv
from torch_geometric.utils import to_dense_batch
from collections import OrderedDict
from Transformer import Encoder
from cross_attn_model import Inner_EncoderLayer, Inter_EncoderLayer
# %%

class ligand_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ligand_GCN, self).__init__()
        self.l_GCN1 = GraphConv(input_dim, hidden_dim)
        self.l_GCN2 = GraphConv(hidden_dim, hidden_dim * 2)
        self.l_GCN3 = GraphConv(hidden_dim * 2, hidden_dim * 2)
        self.l_fc1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.01),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
    def forward(self, x_l, edge_index_l, batch_l):
        # ligand
        x_l = self.l_GCN1(x_l, edge_index_l)
        x_l = self.relu(x_l)
        x_l = self.l_GCN2(x_l, edge_index_l)
        x_l = self.relu(x_l)
        x_l = self.l_GCN3(x_l, edge_index_l)
        x_l = self.relu(x_l)
        x_l_atom = self.l_fc1(x_l)
        x_l = global_add_pool(x_l_atom, batch_l)

        return x_l_atom, x_l

class prot_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(prot_GCN, self).__init__()
        self.l_GCN1 = GraphConv(input_dim, hidden_dim)
        self.l_GCN2 = GraphConv(hidden_dim, hidden_dim * 2)
        self.l_GCN3 = GraphConv(hidden_dim * 2, hidden_dim * 2)
        self.l_fc1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.01),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
    def forward(self, x_l, edge_index_l, batch_l):
        # ligand
        x_l = self.l_GCN1(x_l, edge_index_l)
        x_l = self.relu(x_l)
        x_l = self.l_GCN2(x_l, edge_index_l)
        x_l = self.relu(x_l)
        x_l = self.l_GCN3(x_l, edge_index_l)
        x_l = self.relu(x_l)
        x_l_atom = self.l_fc1(x_l)
        x_l = global_add_pool(x_l_atom, batch_l)

        return x_l_atom, x_l

class pocket_mole_representation(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=128, n_layers=6):
        super(pocket_mole_representation, self).__init__()
        self.prot_encoder = Encoder(src_vocab_size=21, d_model=128, d_ff=256, n_layers=6)
        self.smiles_encoder = Encoder(src_vocab_size=68, d_model=128, d_ff=256, n_layers=6)
        self.prot_GCN = prot_GCN(input_dim=128, hidden_dim=128, output_dim=128)
        self.ligand_GCN = ligand_GCN(input_dim=63, hidden_dim=128, output_dim=128)

        self.InnerCrossAttn = nn.ModuleList([Inner_EncoderLayer() for _ in range(n_layers)])
        self.InterCrossAttn1 = nn.ModuleList([Inter_EncoderLayer() for _ in range(n_layers)])
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, out_dim, bias=False),
        )

    def extract_res(self, x_global, x_trans_mask):
        indices = torch.nonzero(x_trans_mask, as_tuple=True)
        x_part = x_global[indices[0], indices[1]]

        return x_part

    def get_attn_pad_mask(self, seq_q, seq_k):
        '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        '''
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked

        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def forward(self, seq_trans, pock_res, ligand_graph, smiles):
        x_p_seq, x_seq_mask, batch_seq = seq_trans.x, seq_trans.y, seq_trans.batch
        x_p, edge_index_p, batch_p = pock_res.x, pock_res.edge_index, pock_res.batch
        x_l, edge_index_l, edge_attr_l, batch_l = ligand_graph.x, ligand_graph.edge_index, ligand_graph.edge_attr, ligand_graph.batch
        x_l_smiles, batch_smiles = smiles.x, smiles.batch

        x_seq_self_attn, _ = self.prot_encoder(x_p_seq)
        x_poc = self.extract_res(x_seq_self_attn, x_seq_mask)
        x_poc_self_attn, mask_p = to_dense_batch(x_poc, batch_p)

        x_smi_pad, mask_l = to_dense_batch(x_l_smiles, batch_smiles)
        x_smi_self_attn, _ = self.smiles_encoder(x_smi_pad)

        x_p_atom, x_p_global = self.prot_GCN(x_poc, edge_index_p, batch_p)
        x_l_atom, x_l_global = self.ligand_GCN(x_l, edge_index_l, batch_l)

        attn_mask_p = self.get_attn_pad_mask(mask_p.int(), mask_l.int())
        attn_mask_l = self.get_attn_pad_mask(mask_l.int(), mask_p.int())

        for layer in self.InnerCrossAttn:
            x_poc_self_attn, _ = layer(x_poc_self_attn, x_p_global, x_p_global, None)
            x_smi_self_attn, _ = layer(x_smi_self_attn, x_l_global, x_l_global, None)
        for layer in self.InterCrossAttn1:
            x_poc_self_attn, _ = layer(x_poc_self_attn, x_smi_self_attn, x_smi_self_attn, attn_mask_p)
            x_smi_self_attn, _ = layer(x_smi_self_attn, x_poc_self_attn, x_poc_self_attn, attn_mask_l)
        x_poc_cross_attn = global_add_pool(x_poc_self_attn[mask_p], batch_p)
        x_smi_cross_attn = global_add_pool(x_smi_self_attn[mask_l], batch_smiles)

        x_mole = torch.cat([x_poc_cross_attn, x_smi_cross_attn], dim=-1)
        x_mole = self.projection(x_mole)

        return x_mole

class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)

class seq_representation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):  # 3 26 128
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 128)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x

class MHAN(nn.Module):
    def __init__(self, filter_num=128, out_dim=1, block_num=3, vocab_protein_size=21, embedding_size=128):
        super().__init__()
        self.pocket_mole_encoder = pocket_mole_representation(hidden_dim=128, out_dim=128, n_layers=1)
        self.seq_encoder = seq_representation(block_num, vocab_protein_size, embedding_size)
        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(256, out_dim)
        )

    def forward(self, seq_trans, pock_res, seq, ligand_graph, smiles):

        x_p_mole = self.pocket_mole_encoder(seq_trans, pock_res, ligand_graph, smiles)
        seq_x = self.seq_encoder(seq.x)
        x = torch.cat([x_p_mole, seq_x], dim=-1)
        x = self.classifier(x)

        return x
