import torch.nn as nn
import torch

class EmbeddingGenerator(nn.Module):
    def __init__(self, embedding_size_time, embedding_size_variable, embedding_size_sector, input_size, max_seq_length):
        super().__init__()
        self.input_size = input_size
        self.input_dim = max_seq_length//input_size
        self.emb_dim = embedding_size_time * self.input_dim

        self.variable_emb_generator = VariableEmbedding(input_size, embedding_size_variable)
        self.time_2_vec = Time2Vec(input_dim=self.input_dim, embed_dim=self.emb_dim)
        self.sector_emb_generator = VariableEmbedding(input_size, embedding_size_sector)

    def forward(self, sequence, time_index_sequence, variable_index_sequence, sector_index_sequence):
        sect_embedding = self.sector_emb_generator(torch.squeeze(sector_index_sequence).long()).view(sequence.shape[0], sequence.shape[1], -1)
        var_embedding = self.variable_emb_generator(torch.squeeze(variable_index_sequence).long()).view(sequence.shape[0], sequence.shape[1], -1)
        time_embedding = self.time_2_vec(torch.squeeze(time_index_sequence[0, :self.input_dim]).long()).view(1, self.input_dim, self.emb_dim//self.input_dim)
        time_embedding = time_embedding.repeat(sequence.shape[0], self.input_size, 1)
        return torch.cat((sequence, var_embedding, time_embedding,sect_embedding), 2)


class VariableEmbedding(nn.Module):

    def __init__(self, num_variables, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(num_variables, embedding_size)
        nn.init.xavier_uniform_(self.embed.weight)


    def forward(self, sequence):
        return self.embed(sequence)


class Time2Vec(nn.Module):
    def __init__(self, input_dim, embed_dim, act_function=torch.sin):
        assert embed_dim % input_dim == 0
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.embed_bias = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.act_function = act_function

    def forward(self, x):
        if self.enabled:
            x = torch.diag_embed(x)
            x = x.float()
            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            x_affine_0, x_affine_remain = torch.split(
                x_affine, [1, self.embed_dim - 1], dim=-1
            )
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
        else:
            x_output = x
        return x_output
