"""Optional helper to compute image features, so that BRIE can be used on raw image files."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def extract_image_features(
    image_paths: Sequence[str],
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Computes a 1536-dimensional embedding per image with a pre-trained Inception-ResNet-v2.

    The paper feeds BRIE with 1536-dimensional Inception-ResNet-v2 embeddings; this uses
    the same architecture with the ImageNet weights of ``timm`` (the original weights of the
    datasets' embeddings may differ slightly). BRIE works with any feature vectors, so pass
    your own matrix to ``BRIERanker.fit`` if you prefer another extractor.

    Requires the optional dependencies: ``pip install "brie[images]"``. The weights are
    downloaded the first time this is called.

    Returns:
        Array of shape ``(len(image_paths), 1536)``; row ``i`` is image ``i``.
    """
    try:
        import timm
        import torch
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            'extract_image_features needs timm and Pillow: pip install "brie[images]"'
        ) from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("inception_resnet_v2", pretrained=True, num_classes=0)
    model.eval().to(device)
    preprocess = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))

    features = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch = torch.stack(
                [
                    preprocess(Image.open(path).convert("RGB"))
                    for path in image_paths[start : start + batch_size]
                ]
            )
            features.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(features).astype(np.float32)
