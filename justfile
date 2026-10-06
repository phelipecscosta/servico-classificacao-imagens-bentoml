# Windows: receitas executadas pelo PowerShell (Linux/macOS: sh, o padrão do just)
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Carrega variáveis do .env (se existir) em todas as receitas
set dotenv-load := true

# Lista as receitas disponíveis
default:
    @just --list

# Instala o ambiente EXATO do uv.lock (cria o .venv; falha se o lock estiver desatualizado)
setup:
    uv sync --locked

# Testa o modelo isolado, fora do BentoML (diagnóstico)
check-model:
    uv run python scripts/check_model.py

# Sobe o serviço em http://localhost:3000 — Swagger na mesma URL (Ctrl+C para parar)
serve:
    uv run bentoml serve service:ImageClassifier

# Roda os testes automatizados (sobe o serviço em memória; não precisa de 'just serve')
test:
    uv run pytest -v

# Envia as amostras ao serviço rodando (antes, rode 'just serve' em outro terminal)
demo:
    uv run python scripts/demo.py

# COMANDO ÚNICO: instala o ambiente e valida o serviço de ponta a ponta com predições reais
all: setup test


