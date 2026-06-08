from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer


def _tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [x for x in str(value).split() if x]


def build_player_san_documents(
    game_df: pd.DataFrame,
    *,
    max_tokens_per_player: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Collect each player's own SAN moves into one text document."""
    rng = np.random.default_rng(random_state)
    docs: dict[str, list[str]] = {}

    for row in game_df.itertuples(index=False):
        moves = _tokens(getattr(row, "moves_san", None))
        white = getattr(row, "white", None)
        black = getattr(row, "black", None)
        if white:
            docs.setdefault(str(white), []).extend(moves[0::2])
        if black:
            docs.setdefault(str(black), []).extend(moves[1::2])

    records = []
    for player, toks in docs.items():
        if len(toks) > max_tokens_per_player:
            idx = np.sort(rng.choice(len(toks), size=max_tokens_per_player, replace=False))
            toks = [toks[i] for i in idx]
        records.append({"player": player, "san_document": " ".join(toks), "n_san_tokens": len(toks)})
    return pd.DataFrame(records)


@dataclass
class AutoencoderResult:
    features: pd.DataFrame
    vectorizer: HashingVectorizer
    model: object
    loss_history: list[float]


class _TorchAE:
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 256):
        import torch
        self.torch = torch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, latent_dim),
            torch.nn.Linear(latent_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )
        self.latent_start = 4

    def to(self, device: str):
        self.net.to(device)
        return self

    def encode(self, x):
        for layer in list(self.net.children())[:self.latent_start + 1]:
            x = layer(x)
        return x


def _sparse_batches(X, batch_size: int, shuffle: bool, random_state: int):
    rng = np.random.default_rng(random_state)
    idx = np.arange(X.shape[0])
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        batch = X[batch_idx]
        yield batch.toarray().astype("float32") if sparse.issparse(batch) else batch.astype("float32")


def train_san_autoencoder_features(
    game_df: pd.DataFrame,
    *,
    latent_dim: int = 8,
    n_features: int = 4096,
    ngram_range: tuple[int, int] = (1, 3),
    max_tokens_per_player: int = 5000,
    min_tokens_per_player: int = 30,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    device: str | None = None,
) -> AutoencoderResult:
    """
    Derive latent player features from raw SAN strings.

    The model is a lightweight denoising-style baseline: player SAN documents are
    converted into hashed move n-gram vectors, and a small PyTorch autoencoder
    compresses them to ``latent_dim`` features. It is much faster than replaying
    every game with a board and keeps the input purely move-text based.
    """
    import torch

    docs = build_player_san_documents(game_df, max_tokens_per_player=max_tokens_per_player, random_state=random_state)
    docs = docs.query("n_san_tokens >= @min_tokens_per_player").reset_index(drop=True)

    vectorizer = HashingVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    X = vectorizer.transform(docs["san_document"])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(random_state)
    model = _TorchAE(input_dim=n_features, latent_dim=latent_dim).to(device)
    opt = torch.optim.AdamW(model.net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    history: list[float] = []
    model.net.train()
    for epoch in range(epochs):
        losses = []
        for batch in _sparse_batches(X, batch_size=batch_size, shuffle=True, random_state=random_state + epoch):
            xb = torch.from_numpy(batch).to(device)
            pred = model.net(xb)
            loss = loss_fn(pred, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else np.nan)

    model.net.eval()
    latents = []
    with torch.no_grad():
        for batch in _sparse_batches(X, batch_size=batch_size, shuffle=False, random_state=random_state):
            xb = torch.from_numpy(batch).to(device)
            z = model.encode(xb).detach().cpu().numpy()
            latents.append(z)
    Z = np.vstack(latents) if latents else np.empty((0, latent_dim))

    features = pd.DataFrame(Z, columns=[f"ae_{i+1}" for i in range(latent_dim)])
    features.insert(0, "player", docs["player"].to_numpy())
    features.insert(1, "n_san_tokens", docs["n_san_tokens"].to_numpy())
    return AutoencoderResult(features=features, vectorizer=vectorizer, model=model, loss_history=history)


def save_autoencoder_result(result: AutoencoderResult, output_dir: str | Path) -> None:
    import joblib
    import torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.features.to_csv(output_dir / "san_autoencoder_features.csv", index=False)
    pd.Series(result.loss_history, name="loss").to_csv(output_dir / "san_autoencoder_loss.csv", index=False)
    joblib.dump(result.vectorizer, output_dir / "san_hashing_vectorizer.joblib")
    torch.save(result.model.net.state_dict(), output_dir / "san_autoencoder_model.pt")
