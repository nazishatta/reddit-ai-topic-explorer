from pathlib import Path
import numpy as np
from umap import UMAP

source = Path("embeddings/reddit_ai_4000.npy")
output = Path("artifacts/projections")
output.mkdir(parents=True, exist_ok=True)

X = np.load(source)
assert X.shape == (4000, 384)
assert np.isfinite(X).all()

settings = [
    ("umap_5d", 5),
    ("umap_2d", 2)
]

for name, dimensions in settings:
    print(f"\nGenerating {name}...", flush=True)

    reducer = UMAP(
        n_neighbors=15,
        n_components=dimensions,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True
    )

    coordinates = reducer.fit_transform(X)

    assert coordinates.shape == (4000, dimensions)
    assert np.isfinite(coordinates).all()

    path = output / f"{name}.npy"
    np.save(path, coordinates.astype("float32"))

    print("Saved:", path.resolve())
    print("Shape:", coordinates.shape)

print("\nINTERACTIVE PROJECTIONS READY")
