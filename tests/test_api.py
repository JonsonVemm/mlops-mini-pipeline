import pytest


def test_rota_health_retorna_200(client):
    """Garante que a rota de saúde está de pé e respondendo sucesso."""
    response = client.get("/ml/health")
    assert response.status_code == 200


def test_rota_health_retorna_formato_json(client):
    """Garante que a resposta da rota de saúde é um dicionário (JSON)."""
    response = client.get("/ml/health")
    assert isinstance(response.json(), dict)


def test_rota_predict_com_dados_validos_retorna_200(client, payload_fraude):
    """Garante que a API processa uma transação correta e devolve a predição."""
    response = client.post("/ml/predict", json=payload_fraude)
    assert response.status_code == 200
    dados_resposta = response.json()
    assert "prediction" in dados_resposta
    assert "probability" in dados_resposta


def test_rota_predict_com_dados_faltando_retorna_422(client):
    """Garante que o Pydantic bloqueia requisições incompletas e devolve 422."""
    payload_invalido = {
        "hora_transacao": 3,
        "distancia_ultima_compra": 1500.0,
        "tentativas_senha": 4,
        "pais_diferente": 1,
    }
    response = client.post("/ml/predict", json=payload_invalido)
    assert response.status_code == 422


def test_predict_probabilidade_robusta(client, payload_fraude):
    """Garante que a probabilidade é válida sem engessar o valor exato."""
    response = client.post("/ml/predict", json=payload_fraude)
    dados = response.json()
    prob = dados["probability"]

    # ROBUSTO ✅
    # 1. Garante que a probabilidade é um número (float)
    assert isinstance(prob, float)

    # 2. Garante que está dentro da regra matemática (entre 0% e 100%)
    assert 0.0 <= prob <= 1.0


@pytest.mark.parametrize(
    "campo, valor_invalido",
    [
        ("valor_transacao", -50.0),  # Valor não pode ser negativo
        ("hora_transacao", 25),  # O dia só tem 24 horas (0 a 23)
        ("tentativas_senha", -1),  # Não existe tentativa negativa
    ],
)
def test_predict_com_campos_invalidos_retorna_422(
    client, payload_fraude, campo, valor_invalido
):
    """Testa múltiplos campos e valores proibidos usando a mesma função."""

    # Fazemos uma cópia do payload válido que veio da fixture
    payload_hackeado = payload_fraude.copy()

    # Injetamos o valor malicioso no campo da vez
    payload_hackeado[campo] = valor_invalido

    # Dispara contra a API
    response = client.post("/ml/predict", json=payload_hackeado)

    # Garante que o Pydantic bloqueou a requisição na porta
    assert response.status_code == 422
