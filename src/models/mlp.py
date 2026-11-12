"""Ensemble of 5 MLPs trained on the training pairs and retrained each round, with revealed points upweighted.

Uncertainty is the spread of the members' predictions. Members differ in initialization and in their bootstrap resample of the data, so the spread reflects data uncertainty, not only optimization noise.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn


class PairMLP(nn.Module):
    """Predict a response from a cell line and an unordered drug pair."""
    def __init__(
            self,
            n_cells: int,
            n_drugs: int,
            dim: int = 32,
            hidden: int = 128
            ):
        super().__init__()
        self.cell = nn.Embedding(n_cells, dim)
        self.drug = nn.Embedding(n_drugs, dim)
        self.head = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(
            self,
            cell: torch.Tensor,
            drug1: torch.Tensor,
            drug2: torch.Tensor
            ) -> torch.Tensor:
        c, d1, d2 = self.cell(cell), self.drug(drug1), self.drug(drug2)
        x = torch.cat([c, d1 + d2, d1 * d2], dim = -1) # commutative building of this vector given the two drugs
        return self.head(x).squeeze(-1)

class MLPEnsemble:
    def __init__(
            self,
            response: str,
            n_models: int = 5,
            revealed_frac: float = 0.3,
            epochs: int = 20,
            batch_size: int = 512,
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            device: str = "cpu",
            seed: int = 0,
            ):
        self.response = response
        self.n_models = n_models
        self.revealed_frac = revealed_frac
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.seed = seed

    def fit(
            self,
            train: pd.DataFrame,
            revealed: pd.DataFrame,
            ) -> "MLPEnsemble":
        """Train every member from scratch on train + revealed."""
        self.cells_ = sorted(train["CELL_LINE"].unique())
        self.drugs_ = sorted(set(train["DRUG1"]) | set(train["DRUG2"]))
        self.y_mean_ = train[self.response].mean()
        self.y_std_ = train[self.response].std()

        # Fit each ensemble member with a different seed
        self.models_ = [self._fit_member(train, revealed, self.seed + i) for i in range(self.n_models)]

        return self

    @torch.no_grad()
    def predict(
        self,
        candidates: pd.DataFrame
        ) -> tuple[np.ndarray, np.ndarray]:
        x = self._encode(candidates)
        preds = torch.stack([m(*x) for m in self.models_]).cpu().numpy()
        preds = preds * self.y_std_ + self.y_mean_
        return preds.mean(axis=0), preds.std(axis=0)

    def _fit_member(
            self,
            train: pd.DataFrame,
            revealed: pd.DataFrame,
            seed: int
            ) -> PairMLP:
        """Train each member on the existing training data and the revealed data."""
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Resample train and revealed rows separately, so every member sees randomly selected revealed rows
        train_, revealed_ = _bootstrap(train, rng), _bootstrap(revealed, rng)
        data = pd.concat([train_, revealed_], ignore_index=True)
        x = self._encode(data)
        y = torch.as_tensor(
            ((data[self.response] - self.y_mean_) / self.y_std_).to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )
        weights = self._sampling_weights(len(train_), len(revealed_))

        model = PairMLP(len(self.cells_) + 1, len(self.drugs_) + 1).to(self.device)

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.epochs * len(data) // self.batch_size):
            i = torch.multinomial(weights, self.batch_size, replacement=True)
            loss = nn.functional.mse_loss(model(*(t[i] for t in x)), y[i])
            opt.zero_grad()
            loss.backward()
            opt.step()

        return model.eval()

    def _sampling_weights(
            self,
            n_train: int,
            n_revealed: int
            ) -> torch.Tensor:
        """Revealed rows make up `revealed_frac` of each minibatch on average."""
        # Before anything is revealed, every training row gets the same weight.
        if n_revealed == 0:
            return torch.ones(n_train, device=self.device)
        f = self.revealed_frac
        # Each training row is weighted 0.7 / n_train. Together sums to 0.7
        # Each revealed row is weighted 0.3 / n_revealed. Together sums to 0.3
        # For example: Each revealed row after 100 rows are revealed counts ~ (0.3 / 100) / (0.7 / 55,000) = 240x as much as a training row --> upsampling on the revealed rows during training.

        return torch.cat([
            torch.full((n_train,), (1 - f) / n_train),
            torch.full((n_revealed,), f / n_revealed)
        ]).to(self.device)

    def _encode(self, df: pd.DataFrame):
        """Create integer ids for cell line and both drugs; 0 is reserved for unseen values. This allows nn.Embedding to look up the learned vector with a number."""
        def ids(col: str, vocab: list[str]) -> torch.Tensor:
            codes = pd.Categorical(df[col], categories=vocab).codes + 1
            return torch.as_tensor(codes, dtype=torch.long, device=self.device)

        return ids("CELL_LINE", self.cells_), ids("DRUG1", self.drugs_), ids("DRUG2", self.drugs_)

def _bootstrap(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Each ensemble member should differ in the data that is selected for bootstrapping. This function draws len(df) row positions at random with replacement."""
    return df.iloc[rng.integers(0, len(df), len(df))]