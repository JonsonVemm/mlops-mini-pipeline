from fastapi.testclient import TestClient
from main import app

# Cria o cliente fantasma para testar a API
client = TestClient(app)


def test_rota_health_retorna_200():
    """Garante que a rota de saúde está de pé e respondendo sucesso."""
    response = client.get("/ml/health")
    assert response.status_code == 200


def test_rota_health_retorna_formato_json():
    """Garante que a resposta da rota de saúde é um dicionário (JSON)."""
    response = client.get("/ml/health")
    assert isinstance(response.json(), dict)


def test_rota_predict_com_dados_validos_retorna_200():
    """Garante que a API processa uma transação correta e devolve a predição."""
    # O pacote de dados simulando o sistema do banco
    payload_valido = {
        "valor_transacao": 8500.50,
        "hora_transacao": 3,
        "distancia_ultima_compra": 1500.0,
        "tentativas_senha": 4,
        "pais_diferente": 1,
    }

    # Enviamos via POST para a rota do modelo
    response = client.post("/ml/predict", json=payload_valido)

    # Verificações
    assert response.status_code == 200
    dados_resposta = response.json()
    assert "prediction" in dados_resposta
    assert "probability" in dados_resposta


def test_rota_predict_com_dados_faltando_retorna_422():
    """Garante que o Pydantic bloqueia requisições incompletas e devolve 422."""
    # Pacote de dados sem o 'valor_transacao' (que é obrigatório)
    payload_invalido = {
        "hora_transacao": 3,
        "distancia_ultima_compra": 1500.0,
        "tentativas_senha": 4,
        "pais_diferente": 1,
    }

    response = client.post("/ml/predict", json=payload_invalido)

    # O Pydantic deve barrar e retornar o Erro 422
    assert response.status_code == 422
