# ── Estágio 1: Builder (A Fábrica) ──
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .

# Instala os pacotes numa pasta separada (/install) para facilitar a cópia depois
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Estágio 2: Final (A Vitrine) ──
FROM python:3.11-slim
WORKDIR /app

# Copia APENAS os pacotes prontos lá da fábrica (builder)
COPY --from=builder /install /usr/local
COPY . .

# Aplica a nossa regra de segurança do usuário não-root
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]