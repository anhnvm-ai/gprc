"""Transformer encoder baseline (PyTorch)."""
from __future__ import annotations
from dataclasses import dataclass, field
import math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def _device(x="auto"):
    if x == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, : x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, ff_dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=ff_dim, dropout=dropout,
                                               batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
    def forward(self, x):
        z = self.pos(self.input_proj(x))
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)

@dataclass
class TransformerBaseline:
    d_model: int = 64; n_heads: int = 4; n_layers: int = 2; ff_dim: int = 128; dropout: float = 0.1
    lr: float = 5e-4; batch_size: int = 64; epochs: int = 200; patience: int = 15
    device: str = "auto"
    history_: list = field(default_factory=list)

    def _loader(self, X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_tr, y_tr, X_va, y_va):
        dev = _device(self.device)
        self.model_ = TimeSeriesTransformer(X_tr.shape[2], self.d_model, self.n_heads,
                                            self.n_layers, self.ff_dim, self.dropout).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()
        tr, va = self._loader(X_tr, y_tr, shuffle=True), self._loader(X_va, y_va)
        best, bad, best_state = float("inf"), 0, None
        for ep in range(self.epochs):
            self.model_.train()
            for xb, yb in tr:
                xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad()
                loss_fn(self.model_(xb), yb).backward(); opt.step()
            sch.step()
            self.model_.eval(); vloss = 0.0; n = 0
            with torch.no_grad():
                for xb, yb in va:
                    xb, yb = xb.to(dev), yb.to(dev)
                    vloss += loss_fn(self.model_(xb), yb).item() * len(yb); n += len(yb)
            vloss /= max(n, 1); self.history_.append(vloss)
            if vloss < best - 1e-6:
                best, bad, best_state = vloss, 0, {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break
        if best_state is not None: self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        dev = _device(self.device); self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.tensor(X, dtype=torch.float32).to(dev)).cpu().numpy()
