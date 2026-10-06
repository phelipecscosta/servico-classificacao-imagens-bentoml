"""Prepara as imagens de demonstração em samples/ (versionada).

- Fotos próprias (data/raw/, ignorada pelo git): aplica a rotação do EXIF,
  remove TODOS os metadados (inclusive GPS) e reduz para no máximo 800 px.
- Imagem sintética: ruído RGB gerado com semente fixa (reprodutível).
"""

import random
from pathlib import Path

from PIL import Image, ImageOps

RAW_DIR = Path("data/raw")
OUT_DIR = Path("samples")
MAX_SIDE = 800  # px no maior lado: leve para o repositório, bem acima dos 224 px do modelo

# Mapeamento explícito original -> publicado (nada fora desta lista é publicado)
PHOTOS = {
    "01_caneta.jpg": "01_caneta.jpg",
    "02_mesa.jpg": "02_mesa.jpg",
    "03a_ceramica.jpg": "03_ceramica.jpg",
    "03b_trilobita.jpg": "04_trilobita.jpg",
}
SYNTHETIC_NAME = "05_ruido_sintetico.png"


def sanitize_photo(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")  # 1º gira, depois descarta o EXIF
    image.thumbnail((MAX_SIDE, MAX_SIDE))  # reduz mantendo a proporção
    image.save(dst, "JPEG", quality=85)  # sem o parâmetro exif=..., nenhum metadado é gravado


def make_noise(dst: Path, size: int = 512, seed: int = 42) -> None:
    data = random.Random(seed).randbytes(size * size * 3)  # 3 bytes (R, G, B) por pixel
    Image.frombytes("RGB", (size, size), data).save(dst, "PNG")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for src_name, dst_name in PHOTOS.items():
        sanitize_photo(RAW_DIR / src_name, OUT_DIR / dst_name)
    make_noise(OUT_DIR / SYNTHETIC_NAME)

    # Verificação: dimensões, tamanho em disco e quantidade de campos EXIF (deve ser 0)
    for path in sorted(OUT_DIR.iterdir()):
        with Image.open(path) as img:
            print(f"{path.name:<24} {img.width}x{img.height}  "
                  f"{path.stat().st_size / 1024:6.0f} KB  EXIF: {len(img.getexif())} campos")


if __name__ == "__main__":
    main()
