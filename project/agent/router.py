def route(state):
    q = state["question"].lower()

    if "zusammenfassung" in q:
        return "summary"

    if "übersetze" in q:
        return "translate"

    return "rag"