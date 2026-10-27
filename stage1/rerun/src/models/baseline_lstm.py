"""LSTM baseline (PyTorch)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def _device(x: str = "auto") -> torch.device:
    if x == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(x)

class LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x, return_hidden: bool = False):
        out, (h, c) = self.lstm(x)
        last = out[:, -1, :]
        y = self.head(last).squeeze(-1)
        if return_hidden:
            return y, out      # full sequence hidden states (B, L, H) for stage-3 Transformer
        return y

@dataclass
class LSTMBaseline:
    hidden: int = 64; layers: int = 2; dropout: float = 0.2
    lr: float = 1e-3; batch_size: int = 64; epochs: int = 200; patience: int = 15
    device: str = "auto"
    history_: list = field(default_factory=list)

    def _make_loader(self, X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_tr, y_tr, X_va, y_va):
        dev = _device(self.device)
        self.model_ = LSTMNet(X_tr.shape[2], self.hidden, self.layers, self.dropout).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.HuberLoss()
        tr = self._make_loader(X_tr, y_tr, shuffle=True)
        va = self._make_loader(X_va, y_va)
        best, bad, best_state = float("inf"), 0, None
        for ep in range(self.epochs):
            self.model_.train()
            for xb, yb in tr:
                xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb); loss.backward(); opt.step()
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

    def predict(self, X) -> np.ndarray:
        dev = _device(self.device); self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.tensor(X, dtype=torch.float32).to(dev)).cpu().numpy()

    def hidden_states(self, X) -> np.ndarray:
        dev = _device(self.device); self.model_.eval()
        with torch.no_grad():
            _, H = self.model_(torch.tensor(X, dtype=torch.float32).to(dev), return_hidden=True)
            return H.cpu().numpy()
