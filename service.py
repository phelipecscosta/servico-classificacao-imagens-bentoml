"""Serviço BentoML de classificação de imagens com ResNet-50 (ImageNet-1k)."""

import bentoml
import torch
from PIL import ImageOps
from PIL.Image import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "microsoft/resnet-50"
MODEL_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"  # versão fixada do modelo


@bentoml.service(traffic={"timeout": 30})  # requisição que passar de 30 s é abortada
class ImageClassifier:
    def __init__(self) -> None:
        # Roda UMA vez, na subida do servidor: o custo de carregar o modelo não se repete
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model.eval()

    @bentoml.api
    def classify(self, image: Image) -> dict:
        """Recebe uma imagem e devolve as 5 classes mais prováveis do ImageNet."""
        # Fotos de celular vêm "deitadas" com a rotação no EXIF: aplica antes de tudo
        image = ImageOps.exif_transpose(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            probs = self.model(**inputs).logits.softmax(dim=-1)[0]

        top = torch.topk(probs, k=5)
        predictions = [
            {"label": self.model.config.id2label[idx.item()], "score": round(score.item(), 4)}
            for score, idx in zip(top.values, top.indices)
        ]
        return {"predictions": predictions}
