import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_size, output_size,
                 hidden_sizes=[128, 128, 128], bn=True, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        prev = input_size
        for hs in hidden_sizes:
            self.layers.append(nn.Linear(prev, hs))
            self.layers.append(nn.ReLU())
            if bn:
                self.layers.append(nn.BatchNorm1d(hs))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            prev = hs
        self.layers.append(nn.Linear(prev, output_size))

    def forward(self, x, _=None):
        if len(x.shape) > 1:
            x = x.view(-1, self.input_size)
        x = x.float()
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)

# Criteo

num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)


class CriteoMLP(nn.Module):
    """Static intent estimator f_theta for Criteo features."""

    def __init__(self, output_size,
                 hidden_sizes=[256, 256, 128],
                 embedding_size=16, bn=True, dropout=0):
        super().__init__()
        self.input_size = 17
        self.embedding_num = (*cate_bin_size, *num_bin_size)
        self.embedding_size = embedding_size
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, embedding_size) for n in self.embedding_num
        ])
        self.layers = nn.ModuleList()
        prev = len(self.embedding_num) * embedding_size
        for hs in hidden_sizes:
            self.layers.append(nn.Linear(prev, hs))
            self.layers.append(nn.ReLU())
            if bn:
                self.layers.append(nn.BatchNorm1d(hs))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            prev = hs
        self.layers.append(nn.Linear(prev, output_size))

    def forward(self, x, _=None, return_emb=False):
        if len(x.shape) > 1:
            x = x.view(-1, self.input_size)
        x_emb = torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        h = x_emb
        for layer in self.layers[:-1]:
            h = layer(h)
        out = self.layers[-1](h)
        return (out, x_emb) if return_emb else out


class CriteoDXY(nn.Module):
    """Dynamic trajectory estimator g_psi(o_h | x, y) for Criteo."""

    def __init__(self, y_size, d_size, hidden_size=128):
        super().__init__()
        self.x_encoder = CriteoMLP(
            hidden_size, [hidden_size] * 3, bn=True)
        self.d_net = MLP(
            input_size=y_size + hidden_size, output_size=d_size,
            hidden_sizes=[hidden_size] * 3, bn=True)

    def forward(self, x, y):
        x_emb = self.x_encoder(x, return_emb=False)
        return self.d_net(torch.cat([x_emb, y], dim=1))


# Taobao

alimama_bin_size = (1000, 10000, 1000)

class TaobaoMLP(nn.Module):
    """Static intent estimator f_theta for Taobao features."""

    def __init__(self, output_size,
                 hidden_sizes=[256, 256, 128],
                 embedding_size=32, bn=True, dropout=0):
        super().__init__()
        self.input_size = 18
        self.embedding_num = alimama_bin_size
        self.embedding_size = embedding_size
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, embedding_size) for n in self.embedding_num
        ])
        self.layers = nn.ModuleList()
        net_input_size = (len(self.embedding_num) * embedding_size
                          + 2 * 5 * embedding_size + 4 * 5)
        prev = net_input_size
        for hs in hidden_sizes:
            self.layers.append(nn.Linear(prev, hs))
            self.layers.append(nn.ReLU())
            if bn:
                self.layers.append(nn.BatchNorm1d(hs))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            prev = hs
        self.layers.append(nn.Linear(prev, output_size))

    def forward(self, x, _=None, return_emb=False):
        batch_size = x.shape[0]
        if len(x.shape) > 1:
            x = x.view(-1, self.input_size)
        mlp_input = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_hist = x[:, 3:].view(-1, 5, 3)
        mlp_input.append(self.embeddings[1](x_hist[:, :, 0]).view(batch_size, -1))
        mlp_input.append(self.embeddings[2](x_hist[:, :, 1]).view(batch_size, -1))
        mlp_input.append(F.one_hot(x_hist[:, :, 2], num_classes=4).view(batch_size, -1))
        x_emb = torch.cat(mlp_input, dim=1)
        h = x_emb
        for layer in self.layers[:-1]:
            h = layer(h)
        out = self.layers[-1](h)
        return (out, x_emb) if return_emb else out


class TaobaoDXY(nn.Module):
    """Dynamic trajectory estimator g_psi(o_h | x, y) for Taobao."""

    def __init__(self, y_size, d_size, hidden_size=256):
        super().__init__()
        self.x_encoder = TaobaoMLP(
            hidden_size, [hidden_size] * 3, bn=True)
        self.d_net = MLP(
            input_size=y_size + hidden_size, output_size=d_size,
            hidden_sizes=[hidden_size] * 3, bn=True)

    def forward(self, x, y):
        x_emb = self.x_encoder(x, return_emb=False)
        return self.d_net(torch.cat([x_emb, y], dim=1))
