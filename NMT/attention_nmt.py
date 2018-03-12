import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from CONSTANTS import EMBEDDING_SIZE, use_cuda

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, EMBEDDING_SIZE)
        self.rnn = nn.LSTM(EMBEDDING_SIZE, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        c0 = Variable(torch.zeros(hidden.size()))
        embedded = self.embedding(input)
        output = embedded
        output, (hidden, cell) = self.rnn(output, (hidden,c0))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 3, 1)

    def forward(self, decoder_hidden, encoder_output):
        
        
        decoder_hidden_trans = decoder_hidden
        decoder_hidden_expanded = decoder_hidden.expand(1, encoder_output.size(1), decoder_hidden.size(2))
        input_vector = torch.cat((decoder_hidden_expanded, encoder_output), 2)
        output = torch.matmul(input_vector, self.attn.weight.t())
        attn_weights = F.softmax(output, dim=1)
        permuted_encoder_output = encoder_output.permute(0, 2, 1)
        input_context = torch.bmm(permuted_encoder_output, attn_weights).view(1,1,-1)
        return input_context

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, EMBEDDING_SIZE)
        self.rnn = nn.LSTM(self.hidden_size * 2 + EMBEDDING_SIZE, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, input_size)

    def forward(self, input_context, hidden, word):
        c0 = Variable(torch.zeros(hidden.size()))
        embedded = self.embedding(word)
        output = torch.cat((embedded, input_context), 2)
        output, (hidden, cell) = self.rnn(output, (hidden,c0))
        output = self.out(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden