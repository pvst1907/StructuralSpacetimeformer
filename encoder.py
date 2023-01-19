import torch
import torch.nn as nn
from attention import LocalSelfAttention, GlobalSelfAttention, StructuralSelfAttention
from pcoding import EmbeddingGenerator


class Encoder(nn.Module):
    def __init__(self, xformer):
        super().__init__()
        self.src_seq_length = xformer.src_seq_length  # N_w
        self.input_size = xformer.input_size
        self.datetime = xformer.datetime_index
        self.embedding_size_time = xformer.embedding_size_time
        self.embedding_size_variable = xformer.embedding_size_variable
        self.embedding_size_sector = xformer.embedding_size_sector
        self.embedding_size = xformer.embedding_size
        self.sec_list = xformer.sector_list
        self.s_qkv = xformer.s_qkv

        self.context_embedding = EmbeddingGenerator(self.embedding_size_time, self.embedding_size_variable,self.embedding_size_sector, self.input_size, self.src_seq_length)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm1 = nn.BatchNorm1d(self.embedding_size)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.local_attention_layer = LocalSelfAttention(self.input_size, self.src_seq_length, self.embedding_size, self.s_qkv)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm2 = nn.BatchNorm1d(self.embedding_size)
        # Self-Attention Layer Structural (in: (N_w x M) out: (N_w x M))
        self.structural_attention_layer = StructuralSelfAttention(self.input_size, self.src_seq_length, self.embedding_size, self.s_qkv, torch.tensor(self.sec_list))
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm3 = nn.BatchNorm1d(self.embedding_size)
        # Self-Attention Layer Global (in: (N_w x M) out: (N_w x M))
        self.global_attention_layer = GlobalSelfAttention(self.input_size, self.src_seq_length, self.embedding_size, self.s_qkv)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm4 = nn.BatchNorm1d(self.embedding_size)
        # FFN  (in: (N_w x M) out: (N_w x M))
        self.W1 = nn.Linear(self.s_qkv, self.s_qkv)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm5 = nn.BatchNorm1d(self.embedding_size)

    def forward(self, sequence):

        # Flattening
        sequence_flat = torch.unsqueeze(torch.flatten(sequence, 1, 2), dim=2)

        # Time & Variable & Sector Encoding/Embedding
        time_index_sequence = torch.flatten(torch.cumsum(torch.full(sequence.size(), 1), -1), 1, 2) -1
        variable_index_sequence = torch.flatten(torch.cumsum(torch.tile(torch.full((sequence.shape[2], ), 1), (sequence.shape[0], sequence.shape[1], 1)), 1), 1, 2) -1
        sector_index_sequence = torch.cat([torch.ones([sequence.shape[0], sequence.shape[2]]) * i for i in self.sec_list], dim=1)

        embedded_sequence = self.context_embedding(sequence_flat, time_index_sequence, variable_index_sequence, sector_index_sequence)

        # Norm
        normed_sequence = self.norm1(embedded_sequence.transpose(2,1))

        # Local Self Attention
        local_attention = self.local_attention_layer(normed_sequence.transpose(2,1))

        # Norm
        normed_local_attention = self.norm2(local_attention.transpose(2,1)+embedded_sequence.transpose(2,1))

        structural_attention = self.structural_attention_layer(normed_local_attention.transpose(2,1))

        # Norm
        normed_sector_attention = self.norm3(structural_attention.transpose(2, 1) + local_attention.transpose(2, 1))

        # Global Self Attention
        global_attention = self.global_attention_layer(normed_sector_attention.transpose(2,1))

        # Norm
        normed_global_attention = self.norm4(global_attention.transpose(2,1) + structural_attention.transpose(2,1))

        # Linear Layer & ReLU
        encoder_out = nn.ReLU()(self.W1(normed_global_attention.transpose(2,1)))

        # Norm
        normed_encoder_out = self.norm5(encoder_out.transpose(2,1)+global_attention.transpose(2,1))

        return normed_encoder_out.transpose(2,1)
