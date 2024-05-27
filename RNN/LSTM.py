import torch
import torch.nn as nn

class LSTMNet(nn.Module):

    def __init__(self, vocab_size, hidden_size, n_layers, dropout):
        super(LSTMNet, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()

        for i in range(self.n_layers):
            l = {}
            l['Whf'] = nn.Linear(self.hidden_size, self.hidden_size)
            l['Whi'] = nn.Linear(self.hidden_size, self.hidden_size)
            l['WhO'] = nn.Linear(self.hidden_size, self.hidden_size)
            l['Whg'] = nn.Linear(self.hidden_size, self.hidden_size)
            l['Wxf'] = nn.Linear(self.vocab_size if i == 0 else self.hidden_size, self.hidden_size)
            l['Wxi'] = nn.Linear(self.vocab_size if i == 0 else self.hidden_size, self.hidden_size)
            l['WxO'] = nn.Linear(self.vocab_size if i == 0 else self.hidden_size, self.hidden_size)
            l['Wxg'] = nn.Linear(self.vocab_size if i == 0 else self.hidden_size, self.hidden_size)
            self.layers.append(nn.ModuleDict(l))

        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 else None

    def forward(self, x, hidden_states, cell_states):
        cstates = []
        outputs = []

        for i in range(self.n_layers):
            inp = x if i == 0 else outputs[-1]

            f = self.sigmoid(self.layers[i]['Whf'](hidden_states[i]) + self.layers[i]['Wxf'](inp))
            ix = self.sigmoid(self.layers[i]['Whi'](hidden_states[i]) + self.layers[i]['Wxi'](inp))
            o = self.sigmoid(self.layers[i]['WhO'](hidden_states[i]) + self.layers[i]['WxO'](inp))
            g = torch.tanh(self.layers[i]['Whg'](hidden_states[i]) + self.layers[i]['Wxg'](inp))

            cstate = f * cell_states[i] + ix * g

            cstates.append(cstate)

            out = o * torch.tanh(cstate)

            outputs.append(out)

        top_h = outputs[-1]
        if self.dropout_layer:
            top_h = self.dropout_layer(top_h)

        logits = self.decoder(top_h)
        s_logits = logits/0.8
        logs = nn.LogSoftmax(dim=-1)
        out = logs(s_logits)
        return out, outputs, cstates