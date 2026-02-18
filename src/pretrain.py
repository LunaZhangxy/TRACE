"""
Unified pretraining script.

Usage:
  python pretrain.py --dataset criteo
  python pretrain.py --dataset taobao
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import get_data, set_seed
from data import get_train_data_from_batch
from model import CriteoMLP, TaobaoMLP
from completer import CriteoCompleter, TaobaoCompleter



def train_static_intent(dataset_name, pretrain_data, device,
                        epochs=1, lr=1e-3, weight_decay=1e-6, batch_size=4096):

    print("=" * 60)

    x, y, d, y_mask, d_mask, test_mask = (
        pretrain_data["x"], pretrain_data["y"], pretrain_data["d"],
        pretrain_data["y_mask"], pretrain_data["d_mask"],
        pretrain_data["test_mask"])
    x, y, d, y_mask, d_mask = get_train_data_from_batch(
        x, y, d, y_mask, d_mask, test_mask)

    if dataset_name == "criteo":
        f_theta = CriteoMLP(output_size=2).to(device)
        d_nt_idx = 5  # 30-day label (last window)
    else:
        f_theta = TaobaoMLP(output_size=2).to(device)
        d_nt_idx = -1  # last window

    optimizer = optim.Adam(f_theta.parameters(), lr=lr, weight_decay=weight_decay)

    # Use final-window labels as ground truth
    mask = d_mask[:, d_nt_idx]
    if d.dim() == 3:
        label = d[:, d_nt_idx, 0]   # pay action for Taobao, sole action for Criteo
    else:
        label = d[:, d_nt_idx]
    lx = x[mask]
    ly = label[mask]
    N = lx.shape[0]

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        f_theta.train()
        total_loss, cnt = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            bx = lx[idx].to(device)
            by = ly[idx].to(device).long().view(-1)
            logits = f_theta(bx)
            loss = F.nll_loss(F.log_softmax(logits, dim=1), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            cnt += len(idx)
        print(f"  [Epoch {epoch}] f_theta loss = {total_loss / cnt:.6f}")

    return f_theta




def train_completer(dataset_name, pretrain_data, device,
                    epochs=3, lr=1e-3, weight_decay=1e-6, batch_size=4096):
    """
    Train the retrospective trajectory completer q_phi(y | x, xi_{1:k})
    with random horizon truncation (Sec. 3.2, Eq. 10).

    Supports both Criteo (K=1) and Taobao (K=3).
    """

    x = pretrain_data["x"]
    y = pretrain_data["y"]
    d = pretrain_data["d"]
    d_mask_full = pretrain_data["d_mask"]

    # Convert to numpy for manipulation
    for name, arr in [("x", x), ("y", y), ("d", d), ("d_mask", d_mask_full)]:
        if isinstance(arr, torch.Tensor):
            if name == "x":
                x = arr.cpu().numpy()
            elif name == "y":
                y = arr.cpu().numpy()
            elif name == "d":
                d = arr.cpu().numpy()
            else:
                d_mask_full = arr.cpu().numpy()

    N = x.shape[0]
    d_nt = d.shape[1]

    # Random horizon truncation
    h_idx = np.random.randint(low=0, high=d_nt, size=N)
    d_mask = np.zeros_like(d_mask_full, dtype=bool)
    for i in range(N):
        d_mask[i, :h_idx[i] + 1] = True

    x_t = torch.from_numpy(x.astype(np.int64))
    y_t = torch.from_numpy(y.astype(np.float32))
    d_t = torch.from_numpy(d.astype(np.int64))
    m_t = torch.from_numpy(d_mask)

    # Ensure d has 3 dimensions: (N, H, K)
    if d_t.dim() == 2:
        d_t = d_t.unsqueeze(-1)  # (N, H) -> (N, H, 1)

    # Construct dataset-specific completer
    if dataset_name == "criteo":
        completer = CriteoCompleter(
            hidden_size=128,
            horizon_embed_dim=16,
            H=d_nt,
        ).to(device)
    elif dataset_name == "taobao":
        completer = TaobaoCompleter(
            hidden_size=128,
            horizon_embed_dim=16,
            H=d_nt,
        ).to(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    optimizer = optim.Adam(completer.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()
    idx_all = np.arange(N)

    for epoch in range(epochs):
        np.random.shuffle(idx_all)
        completer.train()
        total_loss, cnt = 0.0, 0
        for i in range(0, N, batch_size):
            bi = idx_all[i:i + batch_size]
            bx = x_t[bi].to(device)
            by = y_t[bi].to(device)
            bd = d_t[bi].to(device)
            bm = m_t[bi].to(device)

            prob = completer(bx, bd, bm)
            loss = bce(prob, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bi)
            cnt += len(bi)

            if (i // batch_size) % 100 == 0:
                print(f"  Epoch {epoch}, step {i // batch_size}, "
                      f"loss = {loss.item():.6f}")

        print(f"  [Epoch {epoch}] q_phi BCE = {total_loss / cnt:.6f}")

    return completer



def main():
    parser = argparse.ArgumentParser(
        description="Unified pretraining for TRACE components")
    parser.add_argument("--dataset", type=str, default="criteo",
                        choices=["criteo", "taobao"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs_theta", type=int, default=1,
                        help="Epochs for pretraining")
    parser.add_argument("--epochs_phi", type=int, default=3,
                        help="Epochs for q_phi (completer) training")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = get_data(args.dataset)
    pretrain_data = data["pretrain_dataset"]
    os.makedirs("./pretrain_model", exist_ok=True)

    f_theta = train_static_intent(
        args.dataset, pretrain_data, device, epochs=args.epochs_theta)
    path_theta = f"./pretrain_model/{args.dataset}_pretrain.pt"
    torch.save(f_theta.state_dict(), path_theta)
    print(f"Saved pretrain to {path_theta}\n")

    completer = train_completer(
        args.dataset, pretrain_data, device, epochs=args.epochs_phi)
    path_phi = f"./pretrain_model/{args.dataset}_completer.pt"
    torch.save(completer.state_dict(), path_phi)
    print(f"Saved q_phi to {path_phi}\n")

    print("All pretraining complete.")


if __name__ == "__main__":
    main()
