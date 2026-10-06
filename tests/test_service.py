"""Testes do serviço: verificam o FUNCIONAMENTO (HTTP, contrato, erros), não o acerto do modelo."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from service import MODEL_REVISION, ClassificationResponse, ImageClassifier

SAMPLES = sorted((Path(__file__).parent.parent / "samples").glob("*.*"))
URL = "/classify"


@pytest.fixture(scope="module")
def client():
    # scope="module": o modelo é carregado UMA vez para todos os testes (não um por teste)
    # to_asgi(): o serviço roda em memória, sem servidor e sem porta
    with TestClient(ImageClassifier.to_asgi()) as c:
        yield c


def post_image(client: TestClient, path: Path, top_k: int | None = None):
    """Envia uma imagem como multipart, igual ao curl -F."""
    data = {"top_k": str(top_k)} if top_k is not None else {}
    with path.open("rb") as f:
        return client.post(URL, files={"image": (path.name, f)}, data=data)


# ----------------------------- Casos de sucesso -----------------------------
@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_amostras_respondem_200_com_contrato_valido(client, path):
    resp = post_image(client, path)
    assert resp.status_code == 200
    body = ClassificationResponse.model_validate(resp.json())  # o contrato valida a si mesmo
    assert len(body.predictions) == 5  # top_k padrão
    assert body.model.revision == MODEL_REVISION


def test_predicoes_ordenadas_por_score(client):
    scores = [p["score"] for p in post_image(client, SAMPLES[0]).json()["predictions"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("k", [1, 10])  # valores-limite VÁLIDOS
def test_top_k_respeitado(client, k):
    resp = post_image(client, SAMPLES[0], top_k=k)
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == k


# ------------------------------ Casos de erro ------------------------------
@pytest.mark.parametrize("k", [0, 11])  # valores-limite INVÁLIDOS
def test_top_k_fora_da_faixa_retorna_400(client, k):
    assert post_image(client, SAMPLES[0], top_k=k).status_code == 400


def test_arquivo_nao_imagem_retorna_400(client):
    # Regressão da T2.3: antes devolvia 500
    resp = client.post(URL, files={"image": ("falso.jpg", b"isto nao e uma imagem", "image/jpeg")})
    assert resp.status_code == 400
    assert "não é uma imagem válida" in resp.text


def test_requisicao_sem_imagem_retorna_400(client):
    # (None, valor): campo de formulário comum dentro do multipart, sem arquivo
    resp = client.post(URL, files={"top_k": (None, "3")})
    assert resp.status_code == 400
