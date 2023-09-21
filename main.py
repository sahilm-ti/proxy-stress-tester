import argparse
import logging
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import requests
from dotenv import load_dotenv
from retrying import retry
from tqdm import tqdm

from logger import get_logger
from opensearch_client import OpenSearchClient
from schema import EvaluatedQA

load_dotenv()

logging.getLogger().setLevel(logging.INFO)
logger = get_logger(__name__)

keys_used: dict[str, int] = {}


def get_similarity_request_body(
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


def get_conciseness_request_body(
    question: str, old_answer: str, new_answer: str, openai_model: str
) -> dict:
    return {
        "model": openai_model,
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that compares two different answers to the same question. You are to compare the second answer against the first on the following criterion: how concise the answer is. Classify the content in second answer as compared to the first answer, with one of the following labels: more, less, unchanged. Your answer should be a JSON object with conciseness as the key and the label as the value.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\nFirst answer: {old_answer}\nSecond answer: {new_answer}\n. Make sure your response is a json object with conciseness as the key.",
            },
        ],
        "temperature": 0.7,
    }


def get_contains_code_request_body(
    question: str, old_answer: str, new_answer: str, openai_model: str
) -> dict:
    return {
        "model": openai_model,
        "messages": [
            {
                "role": "system",
                "content": 'You are an AI assistant that evaluates answers to a question.\nYou are provided a question, old answer and a new answer.\nEvaluate whether the answer contains any code (programming language snippets, scripts, etc.).\nYour response should be a JSON object with two items:\n1. "old_answer_code" as the key and the value being either true or false.\n2. "new_answer_code" as the key and the value being either true or false.',
            },
            {
                "role": "user",
                "content": f'Question: {question}\nOld answer: {old_answer}\nNew answer: {new_answer}\n. Make sure your response is a json object with "old_answer_code" and "new_answer_code" as keys.',
            },
        ],
        "temperature": 1,
    }


def prepare_requests(qa_pair: EvaluatedQA, model: str) -> list[dict]:
    prompt_types = [
        get_similarity_request_body,
        get_conciseness_request_body,
        get_contains_code_request_body,
    ]
    return [
        f(
            qa_pair.old_qa_pair.question,
            qa_pair.old_qa_pair.answer,
            qa_pair.new_qa_pair.answer,
            model,
        )
        for f in prompt_types
    ]


def send_requests(
    size: int,
    model: str,
    openai_api_keys: list[str],
    opensearch_client: OpenSearchClient,
    proxy_base_endpoint,
) -> None:
    qa_pairs = opensearch_client.get_qa_pairs(
        size=size,
        query={"bool": {"must": [{"exists": {"field": "evaluation.replay_id"}}]}},
    )
    logger.info(f"Found {len(qa_pairs)} qa pairs.")

    def is_rate_limit_error(exception):
        if "Too many requests" in str(exception):
            print(f"Rate limit error, going to retry")
            return True
        return False

    @retry(
        retry_on_exception=is_rate_limit_error,
        wait_random_min=15000,
        wait_random_max=45000,
        stop_max_attempt_number=10,
    )
    def post(request_body: dict) -> requests.Response:
        openai_api_key = random.choice(openai_api_keys)
        keys_used[openai_api_key] = keys_used.get(openai_api_key, 0) + 1
        request_headers = {
            "callback-token": uuid.uuid4().hex[:8],
            "Authorization": f"Bearer {openai_api_key}",
        }
        result = requests.post(
            f"{proxy_base_endpoint}/chat/completions",
            json=request_body,
            headers=request_headers,
        )
        if result.status_code == 429:
            raise Exception("Too many requests")
        return result

    all_requests = []
    for qa_pair in qa_pairs:
        all_requests.extend(prepare_requests(qa_pair, model))
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(post, req): req for req in all_requests}
        progress_bar = tqdm(total=len(futures), desc="Processing requests")
        for future in as_completed(futures):
            progress_bar.update(1)
            try:
                response = future.result()
                logger.debug(f"Received status_code {response.status_code} from proxy")
                if response.status_code != 200:
                    logger.error(
                        f"Received response ({response.status_code}): {response.text} from proxy"
                    )
            except Exception as e:
                logger.error(f"Exception occurred: {e}")
        progress_bar.close()


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

    openai_api_keys = os.environ["OPENAI_API_KEY"].split(",")
    opensearch_project = os.environ["OPENSEARCH_PROJECT"]
    opensearch_username = os.environ["OPENSEARCH_USERNAME"]
    opensearch_password = os.environ["OPENSEARCH_PASSWORD"]
    proxy_base_endpoint = os.environ["PROXY_BASE_ENDPOINT"]

    client = OpenSearchClient(
        project=opensearch_project,
        name=opensearch_username,
        password=opensearch_password,
    )

    start_time = perf_counter()
    send_requests(
        size=args.size,
        model=args.model,
        openai_api_keys=openai_api_keys,
        opensearch_client=client,
        proxy_base_endpoint=proxy_base_endpoint,
    )
    end_time = perf_counter()
    logger.info(f"Total time taken: {end_time - start_time} seconds")
    logger.info(f"Keys used: {keys_used}")


if __name__ == "__main__":
    main()
