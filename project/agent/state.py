from typing import TypedDict, List

class AgentState(TypedDict):
    question: str
    answer: str
    sources: List[str]