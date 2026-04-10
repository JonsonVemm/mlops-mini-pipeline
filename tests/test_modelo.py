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
