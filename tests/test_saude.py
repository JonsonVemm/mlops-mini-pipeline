def test_pytest_funcionando():
    """Confirma que o pytest encontrou e executou este arquivo."""
    assert 1 + 1 == 2

def test_que_vai_falhar():
    """Este teste falha de propósito."""
    assert 1 + 1 == 3