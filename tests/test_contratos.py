import pytest


@pytest.mark.validacao
def test_contrato_post_predict_sucesso(client, payload_fraude):
    """Garante que o formato de resposta da predição não mude de surpresa."""
    response = client.post("/ml/predict", json=payload_fraude)
    assert response.status_code == 200
    dados = response.json()

    # 1. Verifica se as chaves exatas existem
    campos_obrigatorios = {"prediction", "probability"}
    assert campos_obrigatorios.issubset(dados.keys())

    # 2. Verifica os tipos de dados (O contrato)
    assert isinstance(dados["prediction"], int)
    assert isinstance(dados["probability"], float)

    # 3. Verifica restrições lógicas do negócio
    assert dados["prediction"] in [0, 1]
    assert 0.0 <= dados["probability"] <= 1.0


@pytest.mark.validacao
def test_contrato_erro_422(client):
    """Garante que o erro de validação segue o padrão do FastAPI."""
    # Enviamos um payload vazio para forçar o erro 422
    response = client.post("/ml/predict", json={})
    assert response.status_code == 422
    dados = response.json()

    # FastAPI devolve erros de validação dentro de uma lista chamada "detail"
    assert "detail" in dados
    erros = dados["detail"]
    assert isinstance(erros, list)
    assert len(erros) > 0

    # Cada erro deve dizer onde foi (loc) e o que aconteceu (msg)
    for erro in erros:
        assert "loc" in erro
        assert "msg" in erro
        assert isinstance(erro["loc"], list)
        assert isinstance(erro["msg"], str)


@pytest.mark.validacao
def test_contrato_erro_404(client):
    """Garante que rotas inexistentes devolvem o formato de erro padrão."""
    response = client.get("/rota-fantasma-do-banco")
    assert response.status_code == 404
    dados = response.json()

    assert "detail" in dados
    assert isinstance(dados["detail"], str)
