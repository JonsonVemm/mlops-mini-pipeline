from fastapi import APIRouter
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
import numpy as np

router = APIRouter()

REPO_ID = "YgorReis/fraud-detector-v1"
_model = None


def get_model():
    global _model
    if _model is None:
        # Puxa a função que você testou no 4.1
        from model_utils import load_model

        _model = load_model(REPO_ID)
    return _model


class PredictInput(BaseModel):
    valor_transacao: float = Field(gt=0, description="Valor da transação em reais")
    hora_transacao: int = Field(ge=0, le=23, description="Hora do dia (0-23)")
    distancia_ultima_compra: float = Field(
        ge=0, description="Distância geográfica em km"
    )
    tentativas_senha: int = Field(
        ge=1, description="Tentativas de senha antes da transação"
    )
    pais_diferente: int = Field(
        ge=0, le=1, description="1 se país diferente, 0 se igual"
    )


# Schema de Saída
class PredictOutput(BaseModel):
    prediction: int
    probability: float
    label: str
    model_version: str


@router.post("/predict", response_model=PredictOutput)
async def predict(input: PredictInput):
    model = get_model()

    features = np.array(
        [
            [
                input.valor_transacao,
                input.hora_transacao,
                input.distancia_ultima_compra,
                input.tentativas_senha,
                input.pais_diferente,
            ]
        ]
    )

    # Faz a predição
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    label = "Fraude" if prediction == 1 else "Legítima"

    return PredictOutput(
        prediction=prediction,
        probability=round(probability, 4),
        label=label,
        model_version=REPO_ID,
    )


@router.get("/health")
async def health():
    try:
        model = get_model()
        test_input = np.zeros((1, 5))
        model.predict(test_input)
        model_ok = True
        model_info = REPO_ID
        detail = None
    except Exception as e:
        model_ok = False
        model_info = REPO_ID
        detail = str(e)

    body = {
        "api": "ok",
        "model": "ok" if model_ok else "degraded",
        "model_repo": model_info,
        "detail": detail,
    }

    status_code = 200 if model_ok else 503
    return JSONResponse(content=body, status_code=status_code)
