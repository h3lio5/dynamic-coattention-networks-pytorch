from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from dcn.config import Config

config = Config()


class DCN(nn.Module):
    """
    """

    def __init__(self, weights, mode='train'):

        super(DCN, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(weights)
        #================ Document and Question Encoder==============#
        self.encoder = nn.LSTM(config.embedding_dim,
                               config.hidden_dim, batch_first=True)
        self.ques_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        #================ Temporal fusion BiLSTM=====================#
        self.temporal_fusion = nn.LSTM(
            3*config.hidden_dim, config.hidden_dim, batch_first=True, bidirectional=True)
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
        # Trainable sentinel vector. It allows the model to not attend to any particular word in the input
        self.sentinel_c = nn.Parameter(torch.rand(config.hidden_dim,))
        self.sentinel_q = nn.Parameter(torch.rand(config.hidden_dim,))
        self.dropout = nn.Dropout(0.2)

    def forward(self, context_ids, context_lengths, question_ids, question_lengths, answer_span=None):
        """
        Args:
            context_ids:      batch of ids of context tokens, shape=(batch_size,max_context_len)
            context_lengths:  batch consisting of context token lengths, shape=(batch_size)
            question_ids:     batch of ids of question tokens, shape=(batch_size,max_question_len) 
            question_lengths: batch consisting of question token lengths, shape=(batch_size)
            answer_span:      batch consisting of start and end positions of the answer in 
                              the context, shape=(batch_size,2)
        Returns:
            loss(optional): loss incurred by the model due to incorrect span predictions
            start_indices:  start indices of the answer as predicted by the model
            end_indices:    end indices of the answet as predicted by the model
        """
        batch_size = answer_span.size(0)
        # pack the sequences to reduce unnecessary computations
        # It requires the sentences to be sorted in descending order to take
        # full advantage
        context_lengths, perm_index_C = context_lengths.sort(descending=True)
        context_ids = context_ids[perm_index_C]
        question_lengths, perm_index_Q = question_ids.sort(descending=True)
        question_ids = question_ids[perm_index_Q]

        # get the word embeddings for the context and question tokens
        context_emb = self.dropout(self.word_embedding(context_ids))
        question_emb = self.dropout(self.word_embedding(question_ids))
        # Pack the context and question embeddings
        packed_context = pack_padded_sequence(
            context_emb, lengths=context_lengths, batch_first=True)
        packed_question = pack_padded_sequence(
            question_emb, lengths=question_lengths, batch_first=True)

        #============== Encode the context and question =================#
        # Note: I have used the notations given in the paper for easier understanding.
        packed_context_output, (_) = self.encoder(packed_context)
        D, (_) = pad_packed_sequence(packed_context_output)
        D = D.contiguous()
        packed_question_output, (_) = self.encoder(packed_question)
        Q_intermediate, (_) = pad_packed_sequence(packed_question_output)
        Q_intermediate = Q_intermediate.contiguous()
        # Non-linear projection on question encoding space
        Q = F.tanh(self.ques_projection(Q_intermediate))
        # Append the sentinel vector, shape = B x 1 x l
        sentinel_c = self.sentinel_c.unsqueeze(0).expand(
            batch_size, config.hidden_dim).unsqueeze(1).contiguous()
        sentinel_q = self.sentinel_q.unsqueeze(0).expand(
            batch_size, config.hidden_dim).unsqueeze(1)
        # shape changes to B x m+1 x l
        D = torch.cat((D, sentinel_c), 1)
        # shape changes to B x n+1 x l
        Q = torch.cat((Q, sentinel_q), 1)
        #=========================== Coattention ============================#
        D_t = D.transpose(1, 2)  # B x l x m+1
        # Affinity matrix
        L = torch.bmm(Q, D_t)  # B x n+1 x m+1
        # Attention weights for question
        A_Q = F.softmax(L, dim=1)  # B x n+1 x m+1
        A_Q = A_Q.transpose(1, 2)  # B x m+1 x n+1
        # Summary of the document in light of each word of the question
        C_Q = torch.bmm(D_t, A_Q)  # B x l x n+1
        # Attention weights for document/context
        Q_t = Q.transpose(1, 2)  # B x l x n+1
        A_D = F.softmax(L, dim=2)  # B x n+1 x m+1
        # Summary of previous attention context in light of each word of the document is computed
        C_D = torch.bmm(torch.cat((Q_t, C_Q), 1), A_D)  # B x 2l x m+1
        C_D_t = C_D.transpose(1, 2)  # B x m+1 x 2l
        #======================== Temporal Fusion BiLSTM ======================#
        bilstm_input = torch.cat((D, C_D_t), 2)  # B x m+1 x3l
        # Exclude the sentinel vector from further computation
        bilstm_input = bilstm_input[:, :-1, :]
        packed_bilstm_input = pack_padded_sequence(
            bilstm_input, lengths=context_lengths, batch_first=True)
        packed_U, (_) = self.temporal_fusion(packed_bilstm_input)
        U, (_) = pad_packed_sequence(packed_U)
