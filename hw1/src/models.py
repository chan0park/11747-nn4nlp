import random
import torch


class CNN(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags, init_emb=None):
        super(CNN, self).__init__()
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        if init_emb is None:
            torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        else:
            self.embedding.weight.data.copy_(init_emb)

        self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_size,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.projection_layer = torch.nn.Linear(
            in_features=num_filters, out_features=ntags, bias=True)
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words):
        emb = self.embedding(words)                 # B x nwords x emb_size
        emb = emb.permute(0, 2, 1)                  # B x emb_size x nwords
        h = self.conv_1d(emb)                       # B x num_filters x nwords
        h = h.max(dim=2)[0]                         # B x num_filters
        h = self.relu(h)
        out = self.projection_layer(h)              # B x ntags
        return out
