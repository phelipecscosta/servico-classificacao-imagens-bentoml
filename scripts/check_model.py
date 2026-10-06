"""Teste isolado do modelo ResNet-50, fora do BentoML.

Objetivo: provar que o modelo baixa, carrega e classifica corretamente
antes de colocá-lo dentro da camada de serviço.
"""

import io
import time
import urllib.request

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "microsoft/resnet-50"
# Foto pública do dataset COCO (dois gatos), usada na documentação do Hugging Face
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# Commit exato do repositório no Hugging Face (fixa a versão do modelo, como o uv.lock fixa o código)
MODEL_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"


def main() -> None:
    # 1) Carregar modelo e pré-processador (1ª execução: baixa ~100 MB para o cache do HF)
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()  # modo inferência: desliga comportamentos de treino (ex.: BatchNorm)
    print(f"Modelo carregado em {time.perf_counter() - t0:.1f}s")

    # 2) Baixar a imagem para a memória (urllib é da biblioteca padrão: sem dependência extra)
    with urllib.request.urlopen(IMAGE_URL, timeout=30) as resp:
        image = Image.open(io.BytesIO(resp.read())).convert("RGB")
    print(f"Imagem original: {image.size[0]}x{image.size[1]} pixels")

    # 3) Pré-processamento: redimensiona, recorta, normaliza e converte em tensor
    inputs = processor(images=image, return_tensors="pt")
    print(f"Tensor de entrada: {tuple(inputs['pixel_values'].shape)}  (lote, canais, altura, largura)")

    # 4) Inferência: inference_mode desliga o cálculo de gradientes (mais rápido, menos memória)
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
    print(f"Inferência em {(time.perf_counter() - t0) * 1000:.0f} ms")

    # 5) Pós-processamento: softmax transforma 1000 números em probabilidades que somam 1
    probs = logits.softmax(dim=-1)[0]
    top = torch.topk(probs, k=5)
    print("\nTop-5:")
    for score, idx in zip(top.values, top.indices):
        print(f"  {model.config.id2label[idx.item()]:<35} {score.item():6.2%}")


if __name__ == "__main__":
    main()
