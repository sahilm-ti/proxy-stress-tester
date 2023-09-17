from typing import Any, Optional

from pydantic import BaseModel

class Feedback(BaseModel):
    rating: int
    alternative_answer: Optional[list[str]] = None
    comment: Optional[str] = None


class QAPair(BaseModel):
    id: str
    question: str
    answer: str
    feedback: Optional[Feedback] = None
    version: str
    sources: list[str] = []
    
class EvaluationCheck(BaseModel):
    name: str
    checks: dict[str, Any]

class CheckerFailedErrorSchema(BaseModel):
    error: str
    cause: str
    name: Optional[str] = None


class EvaluatedQA(BaseModel):
    replay_id: str
    old_qa_pair: QAPair
    new_qa_pair: QAPair
    checks: list[EvaluationCheck]
    errors: list[CheckerFailedErrorSchema]
