---
language: pt
tags:
  - sklearn
  - classification
  - fraud-detection
  - mlops
---

# fraud-detector-v1

Modelo de classificação binária para detecção de transações fraudulentas.
Desenvolvido como parte do curso de Ciência de Dados e IA (MLOps) da PUC-SP.

## Uso

```python
from huggingface_hub import hf_hub_download
import joblib

model = joblib.load(hf_hub_download("YgorReis/fraud-detector-v1", "model.pkl"))
# Exemplo de input na ordem exata do treino: 
# valor_transacao, hora_transacao, distancia_ultima_compra, tentativas_senha, pais_diferente
features = [[250.0, 14, 12.5, 1, 0]]
prediction = model.predict(features)
```

## Features de entrada

| Feature                 | Tipo  | Descrição                              |
|-------------------------|-------|----------------------------------------|
| valor_transacao         | float | Valor da transação em reais            |
| hora_transacao          | int   | Hora do dia (0-23)                     |
| distancia_ultima_compra | float | Distância geográfica em km             |
| tentativas_senha        | int   | Tentativas de senha antes da transação |
| pais_diferente          | int   | 1 se país diferente do cadastro        |

## Métricas (test set, 20% dos dados)

- **Precision (fraude):** 1.00
- **Recall (fraude):** 1.00
- **F1 (fraude):** 1.00

## Dependências

- scikit-learn
- joblib
- numpy

## Limitações

Modelo treinado com dados sintéticos criados com regras absolutas, o que gerou métricas de 100% de acerto. Totalmente irrealista para cenários verdadeiros. Não deve ser usado em produção sem retreinamento com dados que possuam ruído e sobreposição de classes.