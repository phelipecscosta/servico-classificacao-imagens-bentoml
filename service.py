"""Serviço BentoML de classificação de imagens com ResNet-50 (ImageNet-1k)."""

import time
from typing import Annotated, Any

import bentoml
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidatorFunctionWrapHandler, WrapValidator
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "microsoft/resnet-50"
MODEL_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"  # versão fixada do modelo


# ---------------------- Contrato de saída (aparece no Swagger) ----------------------
class Prediction(BaseModel):
    label: str = Field(description="Classe do ImageNet-1k", examples=["ballpoint, ballpoint pen, ballpen, Biro"])
    score: float = Field(ge=0.0, le=1.0, description="Confiança do softmax (0 a 1); NÃO é probabilidade de acerto", examples=[0.9876])


class ModelInfo(BaseModel):
    id: str = Field(description="Repositório do modelo no Hugging Face", examples=[MODEL_ID])
    revision: str = Field(description="Commit exato do modelo", examples=[MODEL_REVISION])


class ClassificationResponse(BaseModel):
    model: ModelInfo
    inference_ms: float = Field(description="Tempo de pré-processamento + inferência, em ms", examples=[42.7])
    predictions: list[Prediction] = Field(description="Classes ordenadas da mais para a menos provável")


# ------------- Entrada: imagem decodificada pelo BentoML, com falha virando 400 -------------
def decode_or_400(value: Any, handler: ValidatorFunctionWrapHandler) -> Image.Image:
    """Envolve a decodificação do BentoML: arquivo inválido vira erro de VALIDAÇÃO (400), não 500."""
    try:
        img = handler(value)  # o BentoML abre a imagem aqui (validação original)
        img.load()            # força a leitura completa: detecta arquivo truncado ainda na validação
        return img
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        # ValueError é o sinal que o Pydantic entende como "dado inválido" -> o BentoML responde 400
        raise ValueError(
            "O arquivo enviado não é uma imagem válida ou está corrompido. Envie JPEG, PNG, WEBP ou BMP."
        ) from exc


ValidImage = Annotated[Image.Image, WrapValidator(decode_or_400)]


# ---------------------------------- Serviço ----------------------------------
@bentoml.service(traffic={"timeout": 30})  # requisição que passar de 30 s é abortada
class ImageClassifier:
    def __init__(self) -> None:
        # Roda UMA vez, na subida do servidor: o custo de carregar o modelo não se repete
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model.eval()

    @bentoml.api
    def classify(
        self,
        image: ValidImage,  # imagem enviada pelo cliente, já decodificada e validada
        top_k: Annotated[int, Field(ge=1, le=10, description="Quantas classes devolver (1 a 10)")] = 5,
    ) -> ClassificationResponse:
        """Recebe uma imagem e devolve as top_k classes mais prováveis do ImageNet-1k."""
        t0 = time.perf_counter()

        # Fotos de celular vêm "deitadas" com a rotação no EXIF: aplica antes de tudo
        pil_image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.inference_mode():
            probs = self.model(**inputs).logits.softmax(dim=-1)[0]
        top = torch.topk(probs, k=top_k)

        return ClassificationResponse(
            model=ModelInfo(id=MODEL_ID, revision=MODEL_REVISION),
            inference_ms=round((time.perf_counter() - t0) * 1000, 1),
            predictions=[
                Prediction(label=self.model.config.id2label[idx.item()], score=round(score.item(), 4))
                for score, idx in zip(top.values, top.indices)
            ],
        )
