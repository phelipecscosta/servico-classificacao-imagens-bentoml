"""Classifica todas as imagens de uma pasta (padrão: data/raw).

Uso: escolher empiricamente as imagens de demonstração.
"""

import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "microsoft/resnet-50"
MODEL_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not paths:
        sys.exit(f"Nenhuma imagem {sorted(EXTENSIONS)} encontrada em {folder}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()

    for path in paths:
        with Image.open(path) as img:
            # Aplica a rotação gravada no EXIF ANTES de tudo (fotos de celular vêm "deitadas")
            image = ImageOps.exif_transpose(img).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.inference_mode():
            probs = model(**inputs).logits.softmax(dim=-1)[0]
        ms = (time.perf_counter() - t0) * 1000

        top = torch.topk(probs, k=3)
        print(f"\n{path.name}  ({image.width}x{image.height}, {ms:.0f} ms)")
        for score, idx in zip(top.values, top.indices):
            print(f"  {model.config.id2label[idx.item()]:<35} {score.item():6.2%}")


if __name__ == "__main__":
    main()
