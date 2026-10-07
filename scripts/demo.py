"""Demonstração ao vivo: envia as amostras ao serviço RODANDO e resume as respostas.

Pré-requisito: 'just serve' em outro terminal.
"""

import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:3000"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SAMPLES = sorted(
    p for p in (Path(__file__).parent.parent / "samples").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
)


def main() -> None:
    # 1) Verificação de saúde: /readyz é criado automaticamente pelo BentoML
    try:
        httpx.get(f"{BASE_URL}/readyz", timeout=5).raise_for_status()
    except httpx.HTTPError:
        sys.exit(f"Serviço indisponível em {BASE_URL}. Em outro terminal, rode: just serve")

    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        # 2) Casos de sucesso: top-1 de cada amostra
        print(f"{'amostra':<24} {'HTTP':<5} {'classe (top-1)':<42} {'score':>6} {'ms':>6}")
        for path in SAMPLES:
            with path.open("rb") as f:
                resp = client.post("/classify", files={"image": (path.name, f)}, data={"top_k": "1"})
            body = resp.json()
            pred = body["predictions"][0]
            print(f"{path.name:<24} {resp.status_code:<5} {pred['label'][:40]:<42} "
                  f"{pred['score']:>6.2%} {body['inference_ms']:>6.1f}")

        # 3) Caso de erro: bytes que não são imagem devem gerar 400
        resp = client.post("/classify", files={"image": ("falso.jpg", b"nao e imagem", "image/jpeg")})
        print(f"\n{'erro: não é imagem':<24} {resp.status_code:<5} {resp.text}")


if __name__ == "__main__":
    main()


