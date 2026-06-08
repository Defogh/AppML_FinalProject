from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from funcs.san_autoencoder import train_san_autoencoder_features, save_autoencoder_result

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_DIR = DATA_DIR / "lichess_2017_09_full.csv"
OUT_DIR = DATA_DIR / "san_autoencoder_2"


raw = pd.read_csv(IN_DIR)
result = train_san_autoencoder_features(
    raw,
    latent_dim=8,
    n_features=4096,
    ngram_range=(1, 3),
    max_tokens_per_player=5000,
    min_tokens_per_player=50,
    epochs=20,
    batch_size=256,
)

save_autoencoder_result(result, OUT_DIR)
print({"players": len(result.features), "final_loss": result.loss_history[-1], "out_dir": str(OUT_DIR)})