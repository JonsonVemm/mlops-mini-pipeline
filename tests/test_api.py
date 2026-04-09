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
