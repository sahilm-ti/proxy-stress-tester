import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from logger import get_logger
from opensearch_client import OpenSearchClient
from schema import EvaluatedQA

load_dotenv()

logging.getLogger().setLevel(logging.INFO)
logger = get_logger(__name__)


def get_openai_request_body(
    question: str, old_answer: str, new_answer: str, openai_model: str
) -> dict:
    return {
        "model": openai_model,
        "messages": [
            {
                "role": "system",
                "content": 'You are an AI assistant that compares two answers to the same question.\nCompare how similar the two answers are to each other in the context of the given question, based on the following 3 criteria:\nContent: This refers to whether the same facts, arguments, and conclusions are presented in both answers.\nStructure: The organization of information, including the sequence of ideas or arguments, can also be a point of comparison.\nStyle: This covers the tone, complexity, and specific word choices used in the answers.\nRate similarity for each criteria using exactly one of the following labels: [none, low, medium, high, exact]\nYour answer should be a JSON object with the following structure: \n{"content": <similarity label>, "structure": <similarity label>, "style": <similarity label>, "overall": <similarity label}',
            },
            {
                "role": "user",
                "content": f"Question: {question}\nFirst answer: {old_answer}\nSecond answer: {new_answer}\n.",
            },
        ],
        "temperature": 1,
    }


def send_requests(
    size: int,
    model: str,
    openai_api_key: str,
    opensearch_client: OpenSearchClient,
    proxy_base_endpoint,
) -> None:
    qa_pairs = opensearch_client.get_qa_pairs(
        size=size,
        query={"bool": {"must": [{"exists": {"field": "evaluation.replay_id"}}]}},
    )
    logger.info(f"Found {len(qa_pairs)} qa pairs.")

    def post(qa_pair: EvaluatedQA):
        request_body = get_openai_request_body(
            qa_pair.old_qa_pair.question,
            qa_pair.old_qa_pair.answer,
            qa_pair.new_qa_pair.answer,
            model,
        )
        request_headers = {
            "callback-token": "FAKE_TOKEN",
            "Authorization": f"Bearer {openai_api_key}",
        }
        return requests.post(
            f"{proxy_base_endpoint}/chat/completions",
            json=request_body,
            headers=request_headers,
        )

    with ThreadPoolExecutor() as executor:
        responses = list(tqdm(executor.map(post, qa_pairs), total=len(qa_pairs)))
        for response in responses:
            logger.debug(f"Received status_code {response.status_code} from proxy")


def main():
    parser = argparse.ArgumentParser(
        prog="proxy-tester", description="Sends a bunch of requests to the proxy."
    )
    parser.add_argument("-s", "--size", default=100, type=int)
    parser.add_argument("-m", "--model", default="gpt-3.5-turbo-16k", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    openai_api_key = os.environ["OPENAI_API_KEY"]
    opensearch_project = os.environ["OPENSEARCH_PROJECT"]
    opensearch_username = os.environ["OPENSEARCH_USERNAME"]
    opensearch_password = os.environ["OPENSEARCH_PASSWORD"]
    proxy_base_endpoint = os.environ["PROXY_BASE_ENDPOINT"]

    client = OpenSearchClient(
        project=opensearch_project,
        name=opensearch_username,
        password=opensearch_password,
    )

    send_requests(
        size=args.size,
        model=args.model,
        openai_api_key=openai_api_key,
        opensearch_client=client,
        proxy_base_endpoint=proxy_base_endpoint,
    )


if __name__ == "__main__":
    main()
