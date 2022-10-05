import argparse
import time
import json
import numpy as np
import math
import random

import torch
from torch.nn import Embedding, Parameter, Conv1d, Dropout, Linear
from torch.nn.functional import relu, nll_loss, log_softmax
from torch.nn.utils.rnn import pack_padded_sequence

class Model(torch.nn.Module):
    def __init__(self, general_embedding, domain_embedding, num_labels, dropout_prob, nums_filter, kernel_sizes, nums_words):
        super(Model, self).__init__()
        self.general_embedding = Embedding(general_embedding.shape[0], general_embedding.shape[1])
        self.general_embedding.weight = Parameter(torch.from_numpy(general_embedding), requires_grad=False)
        
        self.domain_embedding = Embedding(domain_embedding.shape[0], domain_embedding.shape[1])
        self.domain_embedding.weight = Parameter(torch.from_numpy(domain_embedding), requires_grad=False)
    
        in_channel_size = general_embedding.shape[1] + domain_embedding.shape[1]
        self.conv1 = Conv1d(in_channel_size, nums_filter[0], kernel_sizes[0], padding=nums_words[0])
        self.conv2 = Conv1d(in_channel_size, nums_filter[0], kernel_sizes[1], padding=nums_words[1])
        self.dropout_prob = Dropout(dropout_prob)

        self.conv3 = Conv1d(nums_filter[1], nums_filter[1], kernel_sizes[2], padding=nums_words[0])
        self.conv4 = Conv1d(nums_filter[1], nums_filter[1], kernel_sizes[2], padding=nums_words[0])
        self.conv5 = Conv1d(nums_filter[1], nums_filter[1], kernel_sizes[2], padding=nums_words[0])
        self.linear_ae = Linear(nums_filter[1], num_labels)

    def forward(self, X_cnt, X_mask, X, y=None):
        X_embedding = torch.cat((self.general_embedding(X), self.domain_embedding(X)), dim=2)
        X_embedding = self.dropout_prob(X_embedding).transpose(1, 2)
        X_conv = relu(torch.cat((self.conv1(X_embedding), self.conv2(X_embedding)), dim=1))
        X_conv = self.dropout_prob(X_conv)
        X_conv = relu(self.conv3(X_conv))
        X_conv = self.dropout_prob(X_conv)
        X_conv = relu(self.conv4(X_conv))
        X_conv = self.dropout_prob(X_conv)
        X_conv = relu(self.conv5(X_conv))
        X_conv = X_conv.transpose(1, 2)
        X_logit = self.linear_ae(X_conv)

        if y:
            X_logit = pack_padded_sequence(X_logit, X_cnt, batch_first=True)
            score = nll_loss(log_softmax(X_logit.data), y.data)
        else:
            X_logit = X_logit.transpose(2, 0)
            score = torch.nn.functional.log_softmax(X_logit).transpose(2, 0)
                
        return score