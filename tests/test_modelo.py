import pytest
import numpy as np
from model_utils import load_model  # A função que você fez na Semana 2

# TODO:
REPO_ID = "YgorReis/fraud-detector-v1"


@pytest.fixture(scope="module")
def modelo():
    """Carrega o modelo uma única vez para economizar rede e tempo."""
    return load_model(REPO_ID)


@pytest.fixture
def amostra_valida():
    """
    Simula o array Numpy final gerado pela sua API antes de ir pro modelo.
    As 5 features: [valor, hora, distancia, tentativas, pais_diferente]
    """
    return np.array([[8500.50, 3, 1500.0, 4, 1]])


@pytest.mark.integracao
def test_modelo_carregado_nao_e_none(modelo):
    """Garante que o download funcionou e o arquivo não está corrompido."""
    assert modelo is not None


@pytest.mark.integracao
def test_modelo_tem_metodos_esperados(modelo):
    """Garante que é um modelo do Scikit-Learn válido."""
    assert hasattr(modelo, "predict")
    assert hasattr(modelo, "predict_proba")


@pytest.mark.integracao
def test_predict_retorna_array_com_formato_correto(modelo, amostra_valida):
    """Garante que o predict devolve 0 ou 1."""
    resultado = modelo.predict(amostra_valida)
    assert resultado.shape == (1,)
    assert resultado[0] in [0, 1]


@pytest.mark.integracao
def test_predict_proba_retorna_probabilidades_validas(modelo, amostra_valida):
    """Garante que a probabilidade é matemática válida (soma = 100%)."""
    probas = modelo.predict_proba(amostra_valida)
    assert probas.shape == (1, 2)  # Duas classes (Normal e Fraude)
    assert abs(probas[0].sum() - 1.0) < 1e-6  # Soma deve ser igual a 1


@pytest.mark.integracao
def test_modelo_distingue_casos_extremos(client):
    """
    Teste de sanidade: garante que uma fraude óbvia tem uma
    probabilidade maior que uma compra normal no dia a dia.
    """
    # Caso 1: A vovó comprando pão na padaria da esquina às 2 da tarde
    caso_tipico = {
        "valor_transacao": 15.50,
        "hora_transacao": 14,
        "distancia_ultima_compra": 2.5,
        "tentativas_senha": 1,
        "pais_diferente": 0,
    }

    # Caso 2: Um hacker na Rússia tentando comprar uma TV de 15 mil reais às 3 da manhã
    caso_suspeito = {
        "valor_transacao": 15000.00,
        "hora_transacao": 3,
        "distancia_ultima_compra": 8000.0,
        "tentativas_senha": 4,
        "pais_diferente": 1,
    }

    # Mandamos os dois casos para a API processar
    resp_tipico = client.post("/ml/predict", json=caso_tipico)
    resp_suspeito = client.post("/ml/predict", json=caso_suspeito)

    assert resp_tipico.status_code == 200
    assert resp_suspeito.status_code == 200

    prob_tipico = resp_tipico.json()["probability"]
    prob_suspeito = resp_suspeito.json()["probability"]

    # A grande validação de negócio: a probabilidade do hacker tem que ser maior que a da vovó!
    assert (
        prob_suspeito > prob_tipico
    ), f"Alerta: O modelo achou a Vovó mais suspeita ({prob_tipico}) que o Hacker ({prob_suspeito})!"


@pytest.mark.integracao
def test_modelo_e_deterministico(client, payload_fraude):
    """O mesmo input deve sempre gerar exatamente a mesma probabilidade."""
    # Como a fixture payload_fraude tá no conftest.py, o pytest puxa ela automático!
    resp_1 = client.post("/ml/predict", json=payload_fraude)
    resp_2 = client.post("/ml/predict", json=payload_fraude)

    # Modelos tradicionais de ML não devem mudar a resposta "do nada" para os mesmos dados
    assert resp_1.json()["prediction"] == resp_2.json()["prediction"]
    assert resp_1.json()["probability"] == resp_2.json()["probability"]
