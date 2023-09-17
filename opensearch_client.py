from typing import Any

from elasticsearch import Elasticsearch

from logger import get_logger
from schema import (
    CheckerFailedErrorSchema,
    EvaluatedQA,
    EvaluationCheck,
    Feedback,
    QAPair,
)

OPENSEARCH_HOST_URL = (
    OPENSEARCH_HOST_URL
) = "https://vpc-ti-shared-e2wxgh4hpjqlvfgbqe472c4maa.us-east-1.es.amazonaws.com"


class OpenSearchClient:
    def __init__(self, project: str, name: str, password: str):
        self.project = project
        self.client = Elasticsearch(
            hosts=[OPENSEARCH_HOST_URL],
            http_auth=(name, password),
            use_ssl=True,
            request_timeout=25,
        )
        self.logger = get_logger(__name__)

    def get_qa_pairs(self, size: int, query: dict[str, Any]) -> list[EvaluatedQA]:
        self.logger.debug(f"Going to fetch {size} questions using {query}")
        response = self.client.search(
            index=f"{self.project}_qa_corpus", body={"size": size, "query": query}
        )
        self.logger.debug(f"Found {response=}")
        evaluated_qa_pairs: list[EvaluatedQA] = []
        for hit in response["hits"]["hits"]:
            id = hit["_id"]
            question = hit["_source"]["question"]
            answer = hit["_source"]["answer"]
            version = hit["_source"]["app_version"]
            feedback = hit["_source"].get("feedback")
            sources = hit["_source"].get("sources", [])
            new_qa_pair = QAPair(
                id=id,
                question=question,
                answer=answer,
                feedback=Feedback(**feedback) if feedback else None,
                version=version,
                sources=sources,
            )
            evaluation = hit["_source"].get("evaluation")
            if evaluation is None:
                raise ValueError("Specified QA pair has not been evaluated")
            replay_id = evaluation["replay_id"]
            checks: list[EvaluationCheck] = []
            check_dictionary = evaluation.get("checks")
            # Some answers may contain checks in the old schema, we don't want to take those
            for key, value in check_dictionary.items():
                if isinstance(value, dict):
                    checks.append(EvaluationCheck(name=key, checks=value))
            errors_list = evaluation.get("errors", [])
            errors: list[CheckerFailedErrorSchema] = [
                CheckerFailedErrorSchema(**error) for error in errors_list
            ]
            old_feedback = evaluation["old_qa"].get("evaluation")
            old_qa_pair = QAPair(
                id=evaluation["old_qa"]["id"],
                question=question,
                answer=evaluation["old_qa"]["answer"],
                feedback=Feedback(**old_feedback) if feedback else None,
                version=evaluation["old_qa"]["app_version"],
                sources=evaluation["old_qa"].get("sources", []),
            )
            evaluated_qa_pairs.append(
                EvaluatedQA(
                    replay_id=replay_id,
                    old_qa_pair=old_qa_pair,
                    new_qa_pair=new_qa_pair,
                    checks=checks,
                    errors=errors,
                )
            )
        return evaluated_qa_pairs
