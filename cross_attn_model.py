import torch
import torch as th
import numpy as np
from torch import nn, Tensor
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, GraphConv
from collections import OrderedDict
# from Transformer import Encoder

#Transformer Parameters
d_model = 128 #Embedding Size
d_ff = 512 #FeedForward dimension
d_k = d_v = 16 #dimension of K(=Q), V
n_layers = 1 #number of Encoder
n_heads = 8 #number of heads in Multi-Head Attention

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        #Q: [batch_size, n_heads, len_q, d_k]
        #K: [batch_size, n_heads, len_k, d_k]
        #V: [batch_size, n_heads, len_v(=len_k), d_v]
        #attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = th.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) #scores:[batch_size, n_heads, len_q, len_k]

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) #Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = th.matmul(attn, V) #[batch_size, n_heads, len_q, d_v]
        return context, attn

class Inner_MultiHeadAttention(nn.Module):
    def __init__(self):
        super(Inner_MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)
        self.LN = nn.LayerNorm(d_model)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        #input_Q: [batch_size, d_model]
        #input_K: [batch_size, d_model]
        #input_V: [batch_size, len_v(=len_k), d_model]
        #attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size, len_q = input_Q, input_Q.size(0), input_Q.size(1)

        #input_K1: [batch_size, len_k, d_model]
        #input_V1: [batch_size, len_k, d_model]
        input_K1 = input_K.unsqueeze(1).repeat(1, len_q, 1)
        input_V1 = input_V.unsqueeze(1).repeat(1, len_q, 1)

        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K1).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V1).view(batch_size, -1, n_heads, d_v).transpose(1, 2) #V:[bs, heads, len_v(=len_k), d_v]

        #attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) #attn_mask:[bs, heads, seq_len, seq_len]

        #context:[batch_size, n_heads, len_q, d_v]
        #attn:[batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v) #context:[bs, len_q, heads*d_v]
        output = self.fc(context) #[batch_size, len_q, d_model]
        return self.LN(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.LN = nn.LayerNorm(d_model)
    def forward(self, inputs):
        # inputs:[batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return self.LN(output + residual) # [batch_size, seq_len, d_model]

class Inner_EncoderLayer(nn.Module):
    def __init__(self):
        super(Inner_EncoderLayer, self).__init__()
        self.enc_self_attn = Inner_MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        '''
        enc_inputs:[batch_size, src_len, d_model]
        enc_self_attn_mask:[batch_size, src_len, src_len]

        enc_outputs:[batch_size, src_len, d_model]
        attn:[batch_size, n_heads, src_len, src_len]
        enc_inputs to same Q,K,V
        '''
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs:[batch_size, src_len, d_model]
        return enc_outputs, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.fc0 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)
        self.LN = nn.LayerNorm(d_model)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        #input_Q: [batch_size, len_q, d_model]
        #input_K: [batch_size, len_k, d_model]
        #input_V: [batch_size, len_v(=len_k), d_model]
        #attn_mask: [batch_size, seq_len, seq_len]
        #batch_size, seq_len, model_len = input_Q.size()
        if attn_mask is not None:
            batch_size, seq_len, model_len = input_Q.size()
            residual_2D = input_Q.view(batch_size*seq_len, model_len)
            residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)
        else:
            residual, batch_size = input_Q, input_Q.size(0)
        '''residual, batch_size = input_Q, input_Q.size(0)'''
        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2) #V:[bs, heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            #attn_mask:[batch_size, n_heads, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        #context:[batch_size, n_heads, len_q, d_v]
        #attn:[batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v) #context:[bs, len_q, heads*d_v]
        output = self.fc(context) #[batch_size, len_q, d_model]
        return self.LN(output + residual), attn

class Inter_EncoderLayer(nn.Module):
    def __init__(self):
        super(Inter_EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        #enc_inputs:[batch_size, src_len, d_model]
        #enc_self_attn_mask:[batch_size, src_len, src_len]

        #enc_outputs:[batch_size, src_len, d_model]
        #attn:[batch_size, n_heads, src_len, src_len]
        #enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) #enc_outputs:[batch_size, src_len, d_model]
        return enc_outputs, attn