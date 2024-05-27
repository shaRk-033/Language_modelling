import torch
import torch.nn as nn

class RecurrentNet(nn.Module):

    def __init__(self, vocab_size, hidden_size, n, dropout):
        super(RecurrentNet, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n = n
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.n):
            layer = {}
            layer['xh'] = nn.Linear(self.vocab_size if i == 0 else self.hidden_size, self.hidden_size)
            layer['hh'] = nn.Linear(self.hidden_size, self.hidden_size)
            self.layers.append(nn.ModuleDict(layer))

        self.Why = nn.Linear(self.hidden_size, self.vocab_size)
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 else None

    def forward(self, x, hidden_states):
        outputs = []
        for i in range(self.n):
            inp = x if i == 0 else outputs[-1]
            if self.dropout_layer and i > 0:
                inp = self.dropout_layer(inp)

            out = torch.tanh(self.layers[i]['hh'](hidden_states[i]) + self.layers[i]['xh'](inp))
            outputs.append(out)

        top_h = outputs[-1]
        if self.dropout_layer:
            top_h = self.dropout_layer(top_h)

        logits = self.Why(top_h)
        s_logits = logits/0.8
        logs = nn.LogSoftmax(dim=-1)
        out = logs(s_logits)
        outputs.append(out)
        return out, outputs


