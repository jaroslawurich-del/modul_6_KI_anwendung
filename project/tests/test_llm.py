from llm.factory import ModelFactory

def test_llm():
    llm = ModelFactory.llm()
    r = llm.invoke("hi")
    assert r is not None