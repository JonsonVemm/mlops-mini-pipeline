import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Cria um TestClient que pode ser usado em qualquer teste."""
    return TestClient(app)


@pytest.fixture
def payload_fraude():
    """Dicionário padrão de fraude para reaproveitar nos testes."""
    return {
        "valor_transacao": 8500.50,
        "hora_transacao": 3,
        "distancia_ultima_compra": 1500.0,
        "tentativas_senha": 4,
        "pais_diferente": 1,
    }
