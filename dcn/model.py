from __future__ import unicode_literals, print_function, division
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dcn.config import Config

config = Config()


class DCN(nn.Module):
    """
    """

    def __init__(self, mode='train'):

        #================ Document and Question Encoder==============#
        self.encoder = nn.LSTM(config.embedding_dim,
                               config.hidden_dim, batch_first=True)
        self.ques_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        #================ Temporal fusion BiLSTM=====================#
        self.temporal_fusion = nn.LSTM(
            config.hidden_dim*2, config.hidden_dim, batch_first=True, bidirectional=True)
        #================ Dynamic Decoder ===========================#
        self.dynamic_decoder = nn.LSTM(
            config.hidden_dim*4, config.hidden_dim, batch_size=True)
        #================= Highway Maxout Network====================#
        self.linear_d = nn.Linear(5*config.hidden_dim, config.hidden_dim)
        self.linear_1 = nn.Linear(
            3*config.hidden_dim, config.hidden_dim*config.maxout_pool_size)
        self.linear_2 = nn.Linear(
            config.hidden_dim, config.hidden_dim*config.maxout_pool_size)
        self.linear_3 = nn.Linear(2*config.hidden_dim, config.maxout_pool_size)

    def forward(self):
        """
        """
        pass
