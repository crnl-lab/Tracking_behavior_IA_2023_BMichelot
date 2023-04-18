import torch
import torch.nn as nn
from torch.autograd import Variable

# Create RNN class
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()

        # parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # creating the layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # one time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out, h0

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMModel, self).__init__()

        self.device = device
        self.directions = 1

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,
                            batch_first=True, bidirectional=(self.directions==2))  # batch_first=True (batch_dim, seq_dim, feature_dim)

        # Readout layer
        self.fc = nn.Linear(self.directions*self.directions*hidden_dim, output_dim)
        self.to(device)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.directions*self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.directions*self.layer_dim, x.size(0), self.hidden_dim).to(self.device)


        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!

        if self.directions==2:
          out = torch.cat((out[:,0,:],out[:,-1,:]), dim=1)
          out = self.fc(out)
        else:
          out = self.fc(out[:,-1,:])

        # out.size() --> 100, 10
        return out, h0

class LSTMModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMModel2, self).__init__()

        self.directions = 1
        self.device = device

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, layer_dim,
                            batch_first=True, bidirectional=(self.directions==2))  # batch_first=True (batch_dim, seq_dim, feature_dim)

        # Readout layer
        self.fc2 = nn.Linear(self.directions*self.directions*hidden_dim, 16)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(16, output_dim)
        self.to(device)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.directions*self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.directions*self.layer_dim, x.size(0), self.hidden_dim).to(self.device)


        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        x = self.fc1(x)
        x = self.relu1(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        #out = self.fc2(out[:, -1, :])
        if self.directions==2:
          out = torch.cat((out[:,0,:],out[:,-1,:]), dim=1)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        return out, h0


class LSTMModel3(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMModel3, self).__init__()

        self.device = device
        self.directions = 1

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.fc1 = nn.Linear(input_dim, int(0.5*hidden_dim))

        # LSTM
        #self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,
        self.lstm = nn.LSTM(int(0.5*hidden_dim), hidden_dim, layer_dim,
                            batch_first=True, bidirectional=(self.directions==2))  # batch_first=True (batch_dim, seq_dim, feature_dim)

        # Readout layer
        self.relu1 = nn.ReLU(inplace=False)
        self.sp = nn.Sigmoid()
        self.fc = nn.Linear(self.directions*self.directions*hidden_dim, output_dim)
        self.to(device)

    def forward(self, x):
        #x = self.relu1(self.fc1(x))
        x = self.fc1(x)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.directions*self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.directions*self.layer_dim, x.size(0), self.hidden_dim).to(self.device)


        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!

        
        # take last output from sequence
        """
        if self.directions==2:
          out = torch.cat((out[:,0,:],out[:,-1,:]), dim=1)
          out = self.fc(out)
        else:
          out = self.fc(out[:,-1,:])
          """
          
        #out = self.relu1(out)
        #out = torch.sum(out, dim=1)
        out = torch.logsumexp(out, dim=1) 
        out = self.fc(out)  # ORIGINAL
        #out = self.sp(self.fc(out))  # NEW
        #out = 1.0-torch.exp(-5*self.fc(out))
        #out = torch.log(1+1.5*self.relu1(self.fc(out)))


        # out.size() --> 100, 10
        return out #, h0


class LSTMModel4(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMModel4, self).__init__()

        self.device = device
        self.directions = 1

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # LSTM
        #self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, layer_dim, #dropout=0.3,
                            batch_first=True, bidirectional=(self.directions==2))  # batch_first=True (batch_dim, seq_dim, feature_dim)

        # Readout layer
        self.relu1 = nn.ReLU(inplace=False)
        #self.fc = nn.Linear(self.directions*self.directions*hidden_dim, output_dim) # BEST
        self.fc = nn.Linear(self.directions*self.directions*hidden_dim, 128)
        self.do = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(128, output_dim)
        self.to(device)

    def forward(self, x):
        #x = self.relu1(self.fc1(x))
        x = self.fc1(x)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.directions*self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.directions*self.layer_dim, x.size(0), self.hidden_dim).to(self.device)


        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!

        
        # take last output from sequence
        """
        if self.directions==2:
          out = torch.cat((out[:,0,:],out[:,-1,:]), dim=1)
          out = self.fc(out)
        else:
          out = self.fc(out[:,-1,:])
          """
          
        #out = self.relu1(out)
        #out = torch.sum(out, dim=1)
        out = torch.logsumexp(out, dim=1)
        out = self.do(self.fc(out))
        out = self.fc2(out)
        # out = self.fc(out) # BEST


        # out.size() --> 100, 10
        return out, h0
