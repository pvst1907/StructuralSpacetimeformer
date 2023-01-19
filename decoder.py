import torch
import torch.nn as nn
from attention import LocalSelfAttention, GlobalSelfAttention, CrossAttention
from pcoding import EmbeddingGenerator


class Decoder(nn.Module):
    def __init__(self, xformer):
        super().__init__()
        self.trg_seq_length = xformer.trg_seq_length  # N_w
        self.src_seq_length = xformer.src_seq_length
        self.output_size = xformer.output_size
        self.input_size = xformer.input_size
        self.embedding_size_time = xformer.embedding_size_time
        self.embedding_size_variable = xformer.embedding_size_variable
        self.embedding_size_sector = xformer.embedding_size_sector
        self.embedding_size = xformer.embedding_size
        self.s_qkv = xformer.s_qkv

        self.target_embedding = EmbeddingGenerator(self.embedding_size_time, self.embedding_size_variable, self.embedding_size_sector, self.output_size, self.trg_seq_length)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm1 = nn.BatchNorm1d(self.embedding_size)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.local_attention_layer = LocalSelfAttention(self.output_size, self.trg_seq_length, self.embedding_size, self.s_qkv, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm2 = nn.BatchNorm1d(self.embedding_size)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.global_attention_layer = GlobalSelfAttention(self.output_size, self.trg_seq_length, self.embedding_size, self.s_qkv, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm3 = nn.BatchNorm1d(self.embedding_size)
        # Cross-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.cross_attention_layer_1 = CrossAttention(self.output_size, self.src_seq_length, self.trg_seq_length, self.embedding_size, self.s_qkv, masked=False)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm4 = nn.BatchNorm1d(self.embedding_size)
        # Cross-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.cross_attention_layer_2 = CrossAttention(self.output_size, self.src_seq_length, self.trg_seq_length, self.embedding_size, self.s_qkv, masked=False)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm5 = nn.BatchNorm1d(self.embedding_size)
        # FFN  (in: (N_w x M) out: (N_w x M))
        self.W1 = nn.Linear(self.s_qkv, self.s_qkv)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm6 = nn.BatchNorm1d(self.embedding_size)
        # Output Layer in: (N_w x M) out: (N_w x 1))
        self.output_layer = nn.Linear(in_features=xformer.s_qkv, out_features=xformer.output_size)

    def forward(self, sequence_trg, sequence_src):
        # Flattening
        sequence_trg_flat = torch.unsqueeze(torch.flatten(sequence_trg, 1, 2), dim=2)

        # No need to flatten the source sequence as it is already flattened
        sequence_src_flat = sequence_src

        # Time & Variable & Sector Encoding/Embedding
        time_index_sequence = torch.flatten(torch.cumsum(torch.full(sequence_trg.size(), 1), 2), 1, 2)
        variable_index_sequence = torch.flatten(torch.cumsum(torch.tile(torch.full((sequence_trg.shape[2],), 1), (sequence_trg.shape[0], sequence_trg.shape[1], 1)), 1), 1, 2)-1
        sector_index_sequence = torch.zeros([sequence_trg.shape[0],sequence_trg_flat.shape[1]])

        embedded_sequence_trg = self.target_embedding(sequence_trg_flat, time_index_sequence, variable_index_sequence,sector_index_sequence)

        # Norm
        normed_sequence_trg = self.norm1(embedded_sequence_trg.transpose(2,1))

        # Local Self Attetion
        local_attention_trg = self.local_attention_layer(normed_sequence_trg.transpose(2,1))

        # Norm
        normed_local_attention_trg = self.norm2(local_attention_trg.transpose(2,1)+embedded_sequence_trg.transpose(2,1))

        # Global Self Attention
        global_attention_trg = self.global_attention_layer(normed_local_attention_trg.transpose(2,1))

        # Norm
        normed_global_attention_trg = self.norm3(global_attention_trg.transpose(2,1)+local_attention_trg.transpose(2,1))

        # Cross Attention 1st Layer
        cross_attention_1 = self.cross_attention_layer_1(normed_global_attention_trg.transpose(2,1), sequence_src_flat)

        # Norm
        normed_cross_attention = self.norm4(cross_attention_1.transpose(2,1)+global_attention_trg.transpose(2,1))

        # Cross Attention 2nd Layer
        cross_attention_2 = self.cross_attention_layer_2(normed_cross_attention.transpose(2,1), sequence_src_flat)

        # Norm
        normed_cross_attention_2 = self.norm5(cross_attention_2.transpose(2,1)+cross_attention_1.transpose(2,1))

        # Linear Layer & ReLU
        decoder_out = nn.ReLU()(self.W1(normed_cross_attention_2.transpose(2,1)))

        # Norm
        normed_decoder_out = self.norm6(decoder_out.transpose(2,1)+cross_attention_2.transpose(2,1))

        # Linear Layer
        return self.output_layer(normed_decoder_out.transpose(2,1))
