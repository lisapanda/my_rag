import getpass
import os
import sys
from pathlib import Path

import pandas as pd
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

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


def evaluate_target(client: Client, target, experiment_prefix: str, correctness_judge) -> None:
    client.evaluate(
        target,
        data=EVAL_DATASET_NAME,
        evaluators=[correctness_judge],
        experiment_prefix=experiment_prefix,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )
    print(f"Started evaluation: {experiment_prefix}")


def main() -> None:
    set_env_if_missing("TAVILY_API_KEY")

    # from rag_design.custom_graph import target as web_search_target
    from rag_design.custom_graph_no_web_search_html_chunks import target as rag_only_target_html
    from rag_design.custom_graph_no_web_search import target as rag_only_target
    from rag_design.models import model

    client = Client()
    create_dataset(client)

    correctness_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=model,
        feedback_key="correctness",
    )

    evaluate_target(
        client,
        rag_only_target_html,
        "rag-eval",
        correctness_judge,
    )
    
    evaluate_target(
        client,
        rag_only_target,
        "rag_only-eval",
        correctness_judge,
    )


if __name__ == "__main__":
    main()
