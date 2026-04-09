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
