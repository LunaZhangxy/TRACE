"""
Retrospective Trajectory Completer  q_phi  (Sec. 3.2).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F



NUM_BIN_SIZE = (64, 16, 128, 64, 128, 64, 512, 512)
CATE_BIN_SIZE = (512, 128, 256, 256, 64, 256, 256, 16, 256)


class CriteoCompleter(nn.Module):
    """
    Retrospective trajectory completer for the Criteo dataset.

    Input:
      x : (B, 17)      static features (discretised indices)
      o : (B, H, 1)    cumulative post-click state per window  (K=1)
      m : (B, H)        visibility mask at the current timestamp

    Output:
      prob : (B,)       q_phi(y=1 | x, xi_{1:k})
    """

    def __init__(
        self,
        hidden_size: int = 256,
        horizon_embed_dim: int = 16,
        H: int = 6,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon_embed_dim = horizon_embed_dim
        self.H = H
        self.embedding_size = embedding_size

        # Feature embeddings
        self.num_features = len(NUM_BIN_SIZE)
        self.cate_features = len(CATE_BIN_SIZE)

        self.cate_embeddings = nn.ModuleList([
            nn.Embedding(CATE_BIN_SIZE[i], embedding_size)
            for i in range(self.cate_features)
        ])
        self.num_embeddings = nn.ModuleList([
            nn.Embedding(NUM_BIN_SIZE[i], embedding_size)
            for i in range(self.num_features)
        ])

        self.x_emb_dim = (self.cate_features + self.num_features) * embedding_size

        # Horizon embedding e_k (encodes the last visible window index)
        self.horizon_embed = nn.Embedding(H, horizon_embed_dim)

        # Trajectory feature: [o * m, m] per window => H * (K + 1)
        # K=1 for Criteo, so each window contributes 2 dims
        self.traj_feat_dim = H * (1 + 1)

        mlp_input_dim = self.x_emb_dim + self.traj_feat_dim + self.horizon_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 2),
        )

    def _embed_x(self, x):
        emb_list = []
        for i in range(self.cate_features):
            emb_list.append(self.cate_embeddings[i](x[:, i]))
        for i in range(self.num_features):
            emb_list.append(self.num_embeddings[i](x[:, self.cate_features + i]))
        return torch.cat(emb_list, dim=1)

    def forward(self, x, o, m):
        """
        x: (B, 17)
        o: (B, H, 1) or (B, H)    cumulative state
        m: (B, H)                  visibility mask
        """
        B = x.shape[0]

        x_emb = self._embed_x(x)

        if o.dim() == 3:
            o = o.squeeze(-1)
        o_float = o.float()
        m_float = m.float()

        # Mask future windows
        o_visible = o_float * m_float
        traj_feat = torch.stack([o_visible, m_float], dim=2).view(B, -1)

        # Last visible horizon index k
        with torch.no_grad():
            idx = torch.arange(self.H, device=m.device).view(1, -1)
            visible_idx = m.bool().long() * idx
            k_idx = visible_idx.max(dim=1).values

        h_emb = self.horizon_embed(k_idx)

        h = torch.cat([x_emb, traj_feat, h_emb], dim=1)
        logits = self.mlp(h)
        prob = F.softmax(logits, dim=1)[:, 1]
        return prob


TAOBAO_BIN_SIZE = (1000, 10000, 1000)


class TaobaoCompleter(nn.Module):
    """
    Retrospective trajectory completer for the Taobao dataset.

    Mirrors the CriteoCompleter design (Sec. 3.2) but adapted for
    Taobao's multi-action trajectories (K=3: cart, fav, pay).

    Input:
      x : (B, 18)      static features (user_id, item_id, item_cate + history)
      o : (B, H, 3)    cumulative post-click state per window  (K=3)
      m : (B, H)        visibility mask at the current timestamp

    Output:
      prob : (B,)       q_phi(y=1 | x, xi_{1:k})
    """

    def __init__(
        self,
        hidden_size: int = 256,
        horizon_embed_dim: int = 16,
        H: int = 5,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon_embed_dim = horizon_embed_dim
        self.H = H
        self.K = 3   # cart, fav, pay
        self.embedding_size = embedding_size

        self.embedding_num = TAOBAO_BIN_SIZE
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, embedding_size) for n in self.embedding_num
        ])

        with torch.no_grad():
            dummy_x = torch.zeros(2, 18, dtype=torch.long)
            x_emb = self._embed_x(dummy_x)
            self.x_emb_dim = x_emb.shape[1]

        # Horizon embedding e_k
        self.horizon_embed = nn.Embedding(H, horizon_embed_dim)

        # Trajectory feature: per window [o_visible (K dims) + mask (1 dim)]
        # => H * (K + 1) total
        self.traj_feat_dim = H * (self.K + 1)

        mlp_input_dim = self.x_emb_dim + self.traj_feat_dim + self.horizon_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 2),
        )

    def _embed_x(self, x):
        """Embed Taobao features: 3 ID embeddings + 5-step history."""
        batch_size = x.shape[0]
        parts = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]

        # History: last 5 interactions  (item_id, item_cate, action_type)
        x_hist = x[:, 3:].view(-1, 5, 3)
        parts.append(self.embeddings[1](x_hist[:, :, 0]).view(batch_size, -1))
        parts.append(self.embeddings[2](x_hist[:, :, 1]).view(batch_size, -1))
        parts.append(
            F.one_hot(x_hist[:, :, 2], num_classes=4).float().view(batch_size, -1)
        )
        return torch.cat(parts, dim=1)

    def forward(self, x, o, m):
        """
        x: (B, 18)
        o: (B, H, 3)    cumulative states (pay, cart, fav)
        m: (B, H)        visibility mask
        """
        B = x.shape[0]

        x_emb = self._embed_x(x)

        o_float = o.float()                              # (B, H, K)
        m_float = m.float().unsqueeze(-1)                 # (B, H, 1)

        # Mask future windows: actions in unseen windows are zeroed
        o_visible = o_float * m_float                     # (B, H, K)
        traj_feat = torch.cat([o_visible, m_float], dim=2)  # (B, H, K+1)
        traj_feat = traj_feat.view(B, -1)                 # (B, H*(K+1))

        # Last visible horizon index k
        with torch.no_grad():
            idx = torch.arange(self.H, device=m.device).view(1, -1)
            visible_idx = m.bool().long() * idx
            k_idx = visible_idx.max(dim=1).values          # (B,)

        h_emb = self.horizon_embed(k_idx)                  # (B, horizon_embed_dim)

        h = torch.cat([x_emb, traj_feat, h_emb], dim=1)
        logits = self.mlp(h)                               # (B, 2)
        prob = F.softmax(logits, dim=1)[:, 1]
        return prob
