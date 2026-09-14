import argparse
import getpass
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import cast

import pandas as pd
from langsmith import Client
from langchain_core.language_models import BaseChatModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import EVAL_DATASET_NAME, EVAL_MAX_CONCURRENCY, GROUND_TRUTH_CSV


def set_env_if_missing(key: str) -> None:
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}: ")


def create_dataset(client: Client) -> None:
    examples = pd.read_csv(GROUND_TRUTH_CSV).to_dict(orient="records")

    if not client.has_dataset(dataset_name=EVAL_DATASET_NAME):
        dataset = client.create_dataset(EVAL_DATASET_NAME)
    else:
        dataset = client.read_dataset(dataset_name=EVAL_DATASET_NAME)

    inputs = [{"input": example["question"]} for example in examples]
    outputs = [{"output": example["answer"]} for example in examples]
    metadata = [
        {
            "source_file": example["source_file"],
            "article": example["article_title"],
            "section": example["section"],
            "page": example["page"],
        }
        for example in examples
    ]

    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_id=dataset.id,
    )
    print(f"Uploaded {len(examples)} examples to dataset '{EVAL_DATASET_NAME}'")


def select_examples(
    client: Client,
    sample_size: int | None = None,
    sample_fraction: float | None = None,
    seed: int | None = None,
):
    if sample_size is None and sample_fraction is None:
        return EVAL_DATASET_NAME

    examples = list(client.list_examples(dataset_name=EVAL_DATASET_NAME))
    requested_size: int
    if sample_size is not None:
        requested_size = sample_size
    else:
        if sample_fraction is None:
            raise ValueError("A sample size or fraction is required.")
        requested_size = max(1, round(len(examples) * sample_fraction))

    if requested_size > len(examples):
        raise ValueError(
            f"Sample size {requested_size} exceeds dataset size {len(examples)}."
        )

    return random.Random(seed).sample(examples, requested_size)


def evaluate_target(
    client: Client,
    target,
    experiment_prefix: str,
    correctness_judge,
    data,
) -> None:
    client.evaluate(
        target,
        data=data,
        evaluators=[correctness_judge],
        experiment_prefix=experiment_prefix,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )
    print(f"Started evaluation: {experiment_prefix}")


def create_correctness_evaluator(judge_model):
    def evaluate(*, inputs, outputs, reference_outputs, **kwargs):
        prompt = f'''Evaluate whether the answer is factually correct and complete.
Return only valid JSON with exactly these fields: {{"score": true or false, "comment": "brief explanation"}}.

Question: {inputs}
Answer: {outputs}
Reference answer: {reference_outputs}
'''
        response = judge_model.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )

        json_match = re.search(r"\{.*\}", str(content), re.DOTALL)
        if json_match is None:
            raise ValueError(f"Judge did not return JSON: {content}")

        result = json.loads(json_match.group())
        if "score" not in result:
            raise ValueError(f"Judge JSON did not contain 'score': {result}")

        return {
            "key": "correctness",
            "score": bool(result["score"]),
            "comment": str(result.get("comment", "")),
        }

    return evaluate


def evaluate_rag_target(
    client: Client,
    sample_size: int | None = None,
    sample_fraction: float | None = None,
    seed: int | None = None,
) -> None:
    set_env_if_missing("TAVILY_API_KEY")

    # from rag_design.custom_graph import target as web_search_target
    # from rag_design.custom_graph_no_web_search_html_chunks import target as rag_only_target_html
    from rag_design.custom_graph_no_web_search_html_recursive_chunks import target as rag_only_target_html_recursive
    from rag_design.models import model

    judge_model = cast(
        BaseChatModel,
        model.model_copy(
            update={
                "max_tokens": 2048,
                "reasoning": {"exclude": True},
            }
        ),
    )
    correctness_judge = create_correctness_evaluator(judge_model)

    data = select_examples(client, sample_size, sample_fraction, seed)

    # # evaluate_target(
    #     client,
    #     rag_only_target_html,
    #     "rag_only_target_html",
    #     correctness_judge,
    # )
    
    evaluate_target(
        client,
        rag_only_target_html_recursive,
        "rag_only_target_html_recursive",
        correctness_judge,
        data,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the evaluation dataset or evaluate a RAG target.")
    parser.add_argument(
        "command",
        choices=("create-dataset", "evaluate-target"),
        help="Workflow to run.",
    )
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--sample-size",
        type=int,
        help="Evaluate this many randomly selected examples.",
    )
    sampling.add_argument(
        "--sample-fraction",
        type=float,
        help="Evaluate this fraction of examples, from 0 to 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be at least 1")
    if args.sample_fraction is not None and not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction must be greater than 0 and at most 1")

    client = Client()
    if args.command == "create-dataset":
        create_dataset(client)
    else:
        evaluate_rag_target(client, args.sample_size, args.sample_fraction, args.seed)


if __name__ == "__main__":
    main()
