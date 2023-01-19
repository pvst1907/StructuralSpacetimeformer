import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_src_trg
from encoder import Encoder
from decoder import Decoder
from train import train_torch
from sklearn import preprocessing
import torch
torch.manual_seed(0)

class SpaceTimeFormer(nn.Module):
    def __init__(self,
                 pred_offset,
                 input_size,
                 output_size,
                 seq_length,
                 sectors_list,
                 datetime_index,
                 embedding_size_time,
                 embedding_size_variable,
                 embedding_size_sector):
        super().__init__()
        self.pred_offset = pred_offset
        self.input_size = input_size
        self.output_size = output_size
        self.seq_length = seq_length
        self.sector_list = sectors_list
        self.datetime_index = datetime_index
        self.src_seq_length = seq_length * input_size
        self.trg_seq_length = (seq_length+(pred_offset-1))*output_size
        self.embedding_size_time = embedding_size_time
        self.embedding_size_variable = embedding_size_variable
        self.embedding_size_sector = embedding_size_sector
        self.embedding_size = 1 + embedding_size_time + embedding_size_variable + embedding_size_sector
        self.s_qkv = self.embedding_size
        self.scores = {}

        self.encoder = Encoder(self)
        self.decoder = Decoder(self)

    def forward(self, source, target):
        source = self.encoder(source)
        output = self.decoder(target, source)
        return output

    def predict(self, source, target_stub, standardize=True):
        if standardize:
            scaler = preprocessing.MinMaxScaler().fit(source)
            source = torch.from_numpy(scaler.transform(source)).float()

            scaler = preprocessing.MinMaxScaler().fit(target_stub)
            target_stub = torch.from_numpy(scaler.transform(target_stub)).float()
        else:
            source = torch.from_numpy(source).float()
            target_stub = torch.from_numpy(target_stub).float()

        target = torch.cat((target_stub, torch.zeros((self.output_size, self.pred_offset))), 1)
        for i in range(self.seq_length-1, self.seq_length + self.pred_offset-1):
            pred = self.forward(torch.unsqueeze(source, dim=0), torch.unsqueeze(target, dim=0))
            target[:, i] = pred[0, i, :]
        return torch.squeeze(pred[0, -self.pred_offset, :]).detach().numpy()

    def start_training(self,
                       source,
                       target,
                       loss,
                       metric,
                       epochs,
                       batch_size,
                       learning_rate,
                       test_size=0.1,
                       standardize=False,
                       verbose=False,
                       plot=False):
        if standardize:
            scaler = preprocessing.MinMaxScaler().fit(source.transpose())
            source = torch.from_numpy(scaler.transform(source.transpose()).transpose()).float()
            scaler = preprocessing.MinMaxScaler().fit(target.transpose())
            target = torch.from_numpy(scaler.transform(target.transpose()).transpose()).float()
        else:
            source = torch.from_numpy(source).float()
            target = torch.from_numpy(target).float()

        # Generate Training Set
        split = int(source.shape[1]*(1-test_size))
        train_iter = load_src_trg(source[:, :split], target[:, :split], self.seq_length, self.pred_offset, batch_size)
        test_iter = load_src_trg(source[:, split:], target[:, split:], self.seq_length, self.pred_offset, batch_size)

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        self.scores['Train'], self.scores['Evaluation'] = train_torch(self, train_iter, test_iter, loss, metric, epochs, encoder_optimizer, decoder_optimizer, verbose=verbose, plot=plot)
