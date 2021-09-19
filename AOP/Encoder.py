# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EdgeEnhancedGCN(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, dep_dim, in_features, out_features, bias=True):
        super(EdgeEnhancedGCN, self).__init__()
        self.dep_dim = dep_dim
        self.in_features = in_features
        self.out_features = out_features

        self.dep_attn = nn.Linear(dep_dim + in_features, out_features)
        self.dep_fc = nn.Linear(dep_dim, out_features)

    def forward(self, text, adj, dep_embed):
        """

        :param text: [batch size, seq_len, feat_dim]
        :param adj: [batch size, seq_len, seq_len]
        :param dep_embed: [batch size, seq_len, seq_len, dep_type_dim]
        :return: [batch size, seq_len, feat_dim]
        """
        batch_size, seq_len, feat_dim = text.shape

        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)
        # [batch size, seq_len, seq_len, feat_dim+pos_dim+dep_dim]
        val_sum = torch.cat([val_us, dep_embed], dim=-1)

        r = self.dep_attn(val_sum)

        p = torch.sum(r, dim=-1)
        mask = (adj == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)

        output = val_us + self.dep_fc(dep_embed)
        output = torch.mul(p_us, output)

        output_sum = torch.sum(output, dim=2)

        return r, output_sum, p


class nESGCN(nn.Module):
    def __init__(self, opt):
        super(nESGCN, self).__init__()
        self.opt = opt
        self.model = nn.ModuleList([EdgeEnhancedGCN(opt.dep_dim, opt.bert_dim,
                                                    opt.bert_dim)
                                    for i in range(self.opt.num_layer)])
        self.dep_embedding = nn.Embedding(opt.dep_num, opt.dep_dim, padding_idx=0)

    def forward(self, x, simple_graph, graph, output_attention=False):

        dep_embed = self.dep_embedding(graph)

        attn_list = []
        for lagcn in self.model:
            r, x, attn = lagcn(x, simple_graph, dep_embed)
            attn_list.append(attn)

        if output_attention is True:
            return x, r, attn_list
        else:
            return x, r


class SpanEncoder(nn.Module):
    def __init__(self, bert, opt):
        super(SpanEncoder, self).__init__()
        self.opt = opt
        self.bert = bert
        self.lagcn = nESGCN(opt)

        self.fc = nn.Linear(opt.bert_dim*2 + opt.pos_dim, opt.bert_dim*2)
        self.bert_dropout = nn.Dropout(opt.bert_dropout)
        self.output_dropout = nn.Dropout(opt.output_dropout)

    def forward(self, input_ids, input_masks, simple_graph, graph, output_attention=False):

        # sequence_output, pooled_output = self.bert(input_ids)
        sequence_output = self.bert(input_ids)[0]
        x = self.bert_dropout(sequence_output)

        lagcn_output = self.lagcn(x, simple_graph, graph, output_attention)

        output = torch.cat((lagcn_output[0], sequence_output), dim=-1)
        output = self.fc(output)
        output = self.output_dropout(output)
        return output, lagcn_output[1]
