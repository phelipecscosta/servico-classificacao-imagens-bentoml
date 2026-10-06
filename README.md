# Serviço de Classificação de Imagens com BentoML

Serviço de inferência que recebe uma imagem por HTTP e devolve as classes mais prováveis,
usando o modelo pré-treinado **ResNet-50** (`microsoft/resnet-50`, ImageNet-1k) servido com **BentoML**.

O foco do projeto é a **camada de serviço**: contrato de API, validação, tratamento de erros,
reprodutibilidade e testes — não o treinamento do modelo.

---

## Sumário

1. [Requisitos](#1-requisitos)
2. [Do `git clone` à primeira predição](#2-do-git-clone-à-primeira-predição)
3. [Contrato da API](#3-contrato-da-api)
4. [Testes e demonstração](#4-testes-e-demonstração)
5. [Estrutura do repositório](#5-estrutura-do-repositório)
6. [O modelo](#6-o-modelo)
7. [Limitações e propostas de melhoria](#7-limitações-e-propostas-de-melhoria)
8. [Problemas comuns](#8-problemas-comuns)
9. [Uso de IA](#9-uso-de-ia)
10. [Licença](#10-licença)

---

## 1. Requisitos

| Ferramenta | Versão testada | Obrigatória? |
|---|---|---|
| Python | 3.11.4 | Sim — mas **não precisa instalar**: o `uv` baixa um Python 3.11 se a máquina não tiver |
| git | 2.50.1 | Sim |
| uv | 0.12.23 | Sim |
| just | 1.58.0 | Não — há comandos equivalentes sem ele |

- **Sistema testado:** Windows 11 (x64). O projeto usa apenas ferramentas multiplataforma (Linux/macOS também suportados).
- **Hardware:** roda em **CPU**; GPU não é necessária nem utilizada.
- **Internet:** necessária na primeira execução (dependências ~300 MB e modelo ~100 MB).

### Instalar o uv

```powershell
# Windows (PowerShell)
winget install --id=astral-sh.uv -e
```

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Instalar o just (opcional)

```powershell
# Windows (PowerShell)
winget install --id=Casey.Just -e
```

```bash
# Qualquer sistema, via uv (sem permissão de administrador)
uv tool install rust-just
```

> Após instalar, **feche e reabra o terminal** (ou o VS Code) para que o PATH seja atualizado.

---

## 2. Do `git clone` à primeira predição

**Tempo aproximado:** 3 a 6 minutos na primeira vez (depende da conexão); segundos nas seguintes.

### Passo 1 — Clonar e instalar o ambiente

```bash
git clone https://github.com/phelipecscosta/servico-classificacao-imagens-bentoml.git
cd servico-classificacao-imagens-bentoml
just setup          # sem just: uv sync --locked
```

O `uv sync --locked` cria o ambiente virtual `.venv` com **exatamente** as versões do `uv.lock`
(e falha com erro claro se o lock estiver inconsistente, em vez de instalar versões diferentes).

### Passo 2 — (Opcional) Validar tudo com um único comando

```bash
just all            # sem just: uv sync --locked && uv run pytest -v
```

Instala o ambiente e executa 12 testes que sobem o serviço **em memória** e fazem predições reais.
Esperado ao final: `12 passed`. Na primeira execução, o modelo (~100 MB) é baixado do Hugging Face.

### Passo 3 — Subir o serviço

```bash
just serve          # sem just: uv run bentoml serve service:ImageClassifier
```

Aguarde a linha `Service ImageClassifier initialized` (~10 a 20 s). O terminal fica ocupado pelo
servidor — **abra um segundo terminal** para os próximos passos. Para parar: `Ctrl+C`.

### Passo 4 — Primeira predição

No **segundo terminal**, na pasta do projeto:

```powershell
# Windows (PowerShell) — use curl.exe, NÃO curl (ver Problemas comuns)
curl.exe -X POST http://localhost:3000/classify -F "image=@samples/01_caneta.jpg"
```

```bash
# Linux / macOS
curl -X POST http://localhost:3000/classify -F "image=@samples/01_caneta.jpg"
```

Resposta esperada (resumida):

```json
{"model":{"id":"microsoft/resnet-50","revision":"34c2154c194f829b11125337b98c8f5f9965ff19"},"inference_ms":74.1,"predictions":[{"label":"ballpoint, ballpoint pen, ballpen, Biro","score":1.0}, ...]}
```

### Alternativa visual: Swagger

Com o serviço no ar, abra **http://localhost:3000** no navegador → `POST /classify` →
**Try it out** → escolha uma imagem de `samples/` → **Execute**.

### Variáveis de ambiente (opcional)

Nenhuma é necessária. Para usar as opcionais, copie o modelo e preencha:

```powershell
Copy-Item .env.example .env     # Windows
```

```bash
cp .env.example .env            # Linux / macOS
```

O `.env` é carregado automaticamente pelas receitas do `just` e **nunca** é versionado.

---

## 3. Contrato da API

### `POST /classify`

**Entrada** — `multipart/form-data`:

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `image` | arquivo | Sim | Imagem JPEG, PNG, WEBP ou BMP. Validada pelo **conteúdo**, não pela extensão |
| `top_k` | inteiro | Não (padrão `5`) | Quantas classes devolver, de `1` a `10` |

**Exemplo de chamada:**

```powershell
curl.exe -X POST http://localhost:3000/classify -F "image=@samples/02_mesa.jpg" -F "top_k=3"
```

**Saída** — `200 OK`, `application/json` (resposta real):

```json
{
  "model": {
    "id": "microsoft/resnet-50",
    "revision": "34c2154c194f829b11125337b98c8f5f9965ff19"
  },
  "inference_ms": 75.0,
  "predictions": [
    {"label": "mouse, computer mouse", "score": 0.5439},
    {"label": "notebook, notebook computer", "score": 0.2388},
    {"label": "laptop, laptop computer", "score": 0.1126}
  ]
}
```

| Campo | Significado |
|---|---|
| `model.id` / `model.revision` | Modelo e commit exato que geraram a resposta (rastreabilidade) |
| `inference_ms` | Tempo de pré-processamento + inferência no servidor |
| `predictions` | Classes do ImageNet-1k, da mais para a menos provável |
| `score` | Confiança do softmax (0 a 1). **Não é probabilidade de acerto** (ver seção 7) |

### Erros

Todo erro causado pela requisição devolve **`400 Bad Request`**. O BentoML usa **dois formatos** de corpo:

**Validação de campos** (`top_k` fora da faixa, `image` ausente):

```json
{"error":"1 validation error for Input","detail":[{"type":"less_than_equal","loc":["top_k"],"msg":"Input should be less than or equal to 10"}]}
```

**Arquivo que não é imagem** (ou imagem corrompida):

```json
[{"error":"O arquivo enviado não é uma imagem válida ou está corrompido. Envie JPEG, PNG, WEBP ou BMP."}]
```

### Endpoints automáticos do BentoML

| Rota | Uso |
|---|---|
| `GET /` | Swagger (documentação interativa) |
| `GET /readyz` | `200` quando o serviço está pronto (modelo carregado) |
| `GET /docs.json` | Especificação OpenAPI |

---

## 4. Testes e demonstração

| Comando | Sem `just` | O que faz |
|---|---|---|
| `just test` | `uv run pytest -v` | 12 testes do serviço em memória (não precisa do servidor) |
| `just demo` | `uv run python scripts/demo.py` | Envia as 5 amostras + 1 caso de erro ao serviço **rodando** |
| `just check-model` | `uv run python scripts/check_model.py` | Testa o modelo isolado, fora do BentoML (diagnóstico) |
| `just` | — | Lista todas as receitas |

Os testes verificam o **funcionamento do serviço** (códigos HTTP, contrato, valores-limite de `top_k`,
erros), e não o acerto do modelo.

**Evidências de execução:** respostas reais do serviço em [`docs/evidencias/`](docs/evidencias/)
(5 casos `200` e 2 casos `400`).

### Imagens de demonstração (`samples/`)

Fotos próprias (sem pessoas e sem dados pessoais) e uma imagem sintética. Metadados EXIF — incluindo
localização GPS — foram removidos por `scripts/prepare_samples.py`. Os originais não são versionados.

| Arquivo | Top-1 | Score | Demonstra |
|---|---|---|---|
| `01_caneta.jpg` | ballpoint | 1.0000 | Caso de sucesso |
| `02_mesa.jpg` | mouse | 0.5439 | Rótulo único numa cena com vários objetos |
| `03_ceramica.jpg` | tray | 0.9800 | Classe inexistente → erro com alta confiança |
| `04_trilobita.jpg` | trilobite | 1.0000 | Composição peculiar do ImageNet |
| `05_ruido_sintetico.png` | hammerhead | 0.0254 | Sem opção "não sei": responde até para ruído |

---

## 5. Estrutura do repositório

```
├── service.py                  # Serviço BentoML: contrato, validação e endpoint /classify
├── justfile                    # Receitas: setup, serve, test, demo, all
├── pyproject.toml              # Dependências declaradas (Python 3.11, torch CPU)
├── uv.lock                     # Versões exatas de todas as dependências (reprodutibilidade)
├── .python-version             # Python 3.11
├── .env.example                # Variáveis de ambiente opcionais (só nomes, sem valores)
├── samples/                    # 5 imagens de demonstração (sem EXIF)
├── scripts/
│   ├── check_model.py          # Diagnóstico: modelo isolado, fora do BentoML
│   ├── classify_images.py      # Classifica uma pasta de imagens (seleção das amostras)
│   ├── prepare_samples.py      # Sanitiza fotos (EXIF/GPS) e gera a imagem sintética
│   └── demo.py                 # Demonstração contra o serviço rodando
├── tests/test_service.py       # 12 testes do serviço
└── docs/
    ├── evidencias/             # Respostas reais do serviço
    └── problemas_e_ajustes.md  # Problemas encontrados, diagnóstico e decisões
```

---

## 6. O modelo

| Item | Descrição |
|---|---|
| **Modelo** | ResNet-50 — `microsoft/resnet-50` no Hugging Face Hub |
| **Versão fixada** | commit `34c2154c194f829b11125337b98c8f5f9965ff19` (garante o mesmo modelo em qualquer máquina) |
| **Arquitetura** | Rede convolucional residual de 50 camadas (He, Zhang, Ren e Sun — Microsoft Research, 2015, *Deep Residual Learning for Image Recognition*) |
| **Treinamento** | Supervisionado no **ImageNet-1k**: ~1,28 milhão de imagens rotuladas em **1.000 classes**, resolução 224×224 |
| **Para quê** | Classificação de imagens em classes fixas; amplamente usado como referência (*benchmark*) e como extrator de características |
| **Licença** | Apache-2.0 (pesos baixados em tempo de execução; não são redistribuídos neste repositório) |
| **Entrada** | Qualquer imagem → redimensionada e **recortada no centro** para 224×224, RGB, normalizada |
| **Saída** | 1.000 valores → softmax → top-k classes |
| **Desempenho local** | ~40 a 80 ms por imagem em CPU (Intel i7-12700H) |

**O que o serviço usa do modelo:** a função original completa — classificação nas 1.000 classes do
ImageNet. Os ajustes feitos foram na **camada de serviço**: aplicação da rotação EXIF (fotos de celular),
conversão para RGB (PNG com transparência, tons de cinza), validação da entrada e limitação do `top_k`.

---

## 7. Limitações e propostas de melhoria

Observadas com as imagens de `samples/` (ver [`docs/evidencias/`](docs/evidencias/)):

| # | Limitação | Evidência | Proposta de melhoria |
|---|---|---|---|
| 1 | **Rótulo único**: uma cena com vários objetos vira uma só resposta | Mesa → `mouse` 54%; a caneca não aparece no top-3 | Modelo de **detecção** (DETR, YOLO), que devolve todos os objetos com suas posições |
| 2 | **Classes fechadas**: só conhece 1.000 classes; o resto é forçado na mais parecida | Prato decorativo → `tray` (bandeja) | Modelo de **vocabulário aberto** (ex.: CLIP), que compara a imagem com rótulos em texto livre |
| 3 | **Confiança não é acerto**: erra com confiança alta | Cerâmica → `tray` com **98%** | **Calibração** (ex.: *temperature scaling*) e não exibir o score como "certeza" |
| 4 | **Sem opção "não sei"** | Ruído → `hammerhead` (2,5%) | **Limiar de confiança**: abaixo dele, responder "desconhecido". Resolve o ruído, mas **não** a cerâmica (98%) — ver #3 |
| 5 | **Rótulos ambíguos** do ImageNet | `notebook` e `laptop` são classes distintas e dividem a probabilidade | Agrupar sinônimos no pós-processamento |
| 6 | **Sensível a mudanças imperceptíveis** | Mesmo foto, só reduzida e recomprimida: `mouse` caiu de 69% para 54% | Testes de robustez; aumento de dados num eventual ajuste fino |
| 7 | **Recorte central**: objetos nas bordas podem ser cortados | Pré-processamento 224×224 central | Preencher (*padding*) em vez de recortar |
| 8 | **Composição do ImageNet**: tem "trilobita" e 120 raças de cães, mas não tem "pessoa" | Trilobita → 100% | Escolher o modelo pelo domínio do problema real |
| 9 | **Apenas CPU, uma imagem por vez** | Decisão de projeto (reprodutibilidade) | Exportar para ONNX; usar o *adaptive batching* do BentoML; GPU opcional |

Problemas de **engenharia** encontrados durante o desenvolvimento (e como foram resolvidos) estão em
[`docs/problemas_e_ajustes.md`](docs/problemas_e_ajustes.md).

---

## 8. Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `curl` no PowerShell dá erro de parâmetro | No PowerShell 5, `curl` é apelido de `Invoke-WebRequest` | Usar **`curl.exe`** |
| `uv` ou `just` "not recognized" | PATH não atualizado após a instalação | Fechar e reabrir o terminal / VS Code |
| `just demo` diz "Serviço indisponível" | O servidor não está rodando | `just serve` em outro terminal |
| Porta 3000 ocupada | Outro processo usa a porta | `uv run bentoml serve service:ImageClassifier --port 3001` |
| Primeira execução lenta | Download do modelo (~100 MB) | Só na primeira vez; depois vem do cache |
| Aviso sobre *symlinks* do Hugging Face | Windows sem "Developer Mode" | Inofensivo; para silenciar, `HF_HUB_DISABLE_SYMLINKS_WARNING=1` no `.env` |
| Aviso de requisições sem `HF_TOKEN` | Download anônimo | Inofensivo; opcionalmente definir `HF_TOKEN` no `.env` |

---

## 9. Uso de IA

**Ferramenta:** Claude (Anthropic), modelo Claude Opus 5.5, via claude.ai.

**O que foi pedido:** orientação técnica e didática, etapa por etapa, para construir o serviço:
estrutura do projeto e boas práticas de MLOps; escolha do modelo; configuração do `uv` (versões fixadas,
PyTorch CPU); código do serviço BentoML, do contrato Pydantic, dos testes, dos scripts e do `justfile`;
interpretação de saídas e erros; e rascunho deste README.

**Avaliação crítica:**

<!-- PREENCHER COM A SUA AVALIAÇÃO. Perguntas-guia:
- Onde a ferramenta acertou e economizou tempo?
- Onde errou ou precisou de correção? (ex.: contagem de testes prevista como 13, real 12;
  previsão de alta confiança para a imagem de ruído, que veio com 2,5%)
- Onde ela desviou do foco e você precisou redirecionar? (ex.: investigação de tempo de rede)
- O que você verificou por conta própria em vez de aceitar a resposta?
- O que você aprendeu e conseguiria refazer sem a ferramenta?
-->

---

## 10. Licença

Código sob licença **MIT** (ver [`LICENSE`](LICENSE)).
O modelo ResNet-50 (`microsoft/resnet-50`) é distribuído pela Microsoft sob licença **Apache-2.0**.
