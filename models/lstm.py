# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from  torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = nn.LSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        x = pack_padded_sequence(x,x_len,batch_first=True,enforce_sorted=False)
        seq,_ = self.lstm(x)
        ho,lens = pad_packed_sequence(seq,batch_first=True)
        # hn = torch.index_select(ho,0,lens-1)
        hn = None
        for i in range(len(lens)):
            if hn is None:
                hn = ho[i,lens[i]-1,:]
            else:
                hn = torch.cat((hn,ho[i,lens[i]-1,:]),0)
        hn = hn.view(len(lens),-1)
        # _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(hn)
        return out

# from layers.dynamic_rnn import DynamicLSTM
# import torch
# import torch.nn as nn


# class LSTM(nn.Module):
#     def __init__(self, embedding_matrix, opt):
#         super(LSTM, self).__init__()
#         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
#         self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

#     def forward(self, inputs):
#         text_raw_indices = inputs[0]
#         x = self.embed(text_raw_indices)
#         x_len = torch.sum(text_raw_indices != 0, dim=-1)
#         _, (h_n, _) = self.lstm(x, x_len)
#         out = self.dense(h_n[0])
#         return out