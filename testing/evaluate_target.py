"""Run and score the configured RAG target with LangSmith.

This module contains only the evaluation workflow. Use
``create_dataset.py`` separately when the LangSmith dataset must be created
from the ground-truth CSV.
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from langchain_core.language_models import BaseChatModel
from langsmith import Client

# Resolve the repository root so this script can import project modules when
# launched from outside the repository directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    CHROMA_HTML_COLLECTION_NAME,
    EVAL_DATASET_NAME,
    EVAL_MAX_CONCURRENCY,
)

from rag_design.models import model
from rag_design.custom_graph_no_web_search_html_recursive_chunks import (
    target as target,
)


def select_examples(
    client: Client,
    sample_size: int | None = None,
    sample_fraction: float | None = None,
    seed: int | None = None,
):
    """Choose the full dataset or return a reproducible random subset.

    When neither sampling option is supplied, returning the dataset name lets
    LangSmith resolve and stream the complete dataset itself. When sampling is
    requested, this function materializes the examples and returns exactly the
    requested number of randomly selected records.
    """

    # No sampling flags means that the evaluator should use every dataset
    # example through the dataset name rather than creating a local list.
    if sample_size is None and sample_fraction is None:
        return EVAL_DATASET_NAME

    # Materialize the dataset only when a local sample must be selected.
    examples = list(client.list_examples(dataset_name=EVAL_DATASET_NAME))
    requested_size: int
    if sample_size is not None:
        # An explicit count takes precedence over fraction-based sampling.
        requested_size = sample_size
    else:
        if sample_fraction is None:
            raise ValueError("A sample size or fraction is required.")
        # Round the requested fraction and keep at least one example so a
        # valid, very small fraction does not silently produce an empty run.
        requested_size = max(1, round(len(examples) * sample_fraction))

    # random.sample cannot select more records than exist, so fail with a
    # clearer message before attempting the sample.
    if requested_size > len(examples):
        raise ValueError(
            f"Sample size {requested_size} exceeds dataset size {len(examples)}."
        )

    # Use a private Random instance so the optional seed makes this selection
    # reproducible without changing Python's process-wide random state.
    return random.Random(seed).sample(examples, requested_size)


def evaluate_target(
    client: Client,
    target,
    experiment_prefix: str,
    correctness_judge,
    data,
    metadata: dict,
) -> None:
    """Start a LangSmith evaluation for one RAG target."""

    # LangSmith executes the target over the selected data, applies the judge,
    # attaches metadata to the experiment, and limits simultaneous evaluations.
    client.evaluate(
        target,
        data=data,
        evaluators=[correctness_judge],
        experiment_prefix=experiment_prefix,
        metadata=metadata,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )

    # LangSmith runs evaluations asynchronously, so this confirms submission
    # rather than claiming that the evaluation has finished.
    print(f"Started evaluation: {experiment_prefix}")


def create_correctness_evaluator(judge_model):
    """Build a LangSmith evaluator backed by ``judge_model``."""

    def evaluate(*, inputs, outputs, reference_outputs, **kwargs):
        """Score one generated answer against its reference answer."""

        # Require a small, fixed JSON schema so the judge response is easy to
        # validate and compatible with LangSmith scores.
        prompt = f'''Evaluate whether the answer is factually correct and complete.
Return only valid JSON with exactly these fields: {{"score": true or false, "comment": "brief explanation"}}.

Question: {inputs}
Answer: {outputs}
Reference answer: {reference_outputs}
'''

        # The judge model may return plain text or structured content blocks.
        response = judge_model.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            # Normalize structured blocks into one string before JSON parsing.
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )

        # Extract the JSON object even if the model surrounds it with prose.
        json_match = re.search(r"\{.*\}", str(content), re.DOTALL)
        if json_match is None:
            raise ValueError(f"Judge did not return JSON: {content}")

        # Require the score field that determines the evaluation result.
        result = json.loads(json_match.group())
        if "score" not in result:
            raise ValueError(f"Judge JSON did not contain 'score': {result}")

        # Normalize booleans and common textual boolean values explicitly.
        # In Python, bool("false") is True, so calling bool() directly on a
        # model-returned string would silently invert a failed evaluation.
        raw_score = result["score"]
        if isinstance(raw_score, bool):
            score = raw_score
        elif isinstance(raw_score, str):
            normalized_score = raw_score.strip().lower()
            if normalized_score in {"true", "yes", "1"}:
                score = True
            elif normalized_score in {"false", "no", "0"}:
                score = False
            else:
                raise ValueError(f"Judge returned an invalid score: {raw_score}")
        else:
            score = bool(raw_score)

        # Return the result in LangSmith's evaluator format.
        return {
            "key": "correctness",
            "score": score,
            "comment": str(result.get("comment", "")),
        }

    # Return the closure so LangSmith can call it once per example.
    return evaluate


def evaluate_rag_target(
    client: Client,
    rag_target,
    rag_model: BaseChatModel,
    collection_name: str,
    sample_size: int | None = None,
    sample_fraction: float | None = None,
    seed: int | None = None,
    run_label: str = "manual",
) -> None:
    """Evaluate a RAG target and record its model and collection settings."""

    # Copy the application model's configuration for judging, but cap the
    # response length and disable reasoning output for concise JSON results.
    judge_model = cast(
        BaseChatModel,
        rag_model.model_copy(
            update={
                "max_tokens": 2048,
                "reasoning": {"exclude": True},
            }
        ),
    )
    correctness_judge = create_correctness_evaluator(judge_model)
    model_name = str(getattr(rag_model, "model_name", type(rag_model).__name__))

    # Resolve full-dataset versus sampled evaluation data from the CLI flags.
    data = select_examples(client, sample_size, sample_fraction, seed)

    # Give every experiment a unique, readable name while preserving the
    # user-provided label for comparing different fixes or configurations.
    started_at = datetime.now(timezone.utc)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_label).strip("-") or "manual"
    experiment_prefix = f"rag_only_target_html_recursive_{safe_label}_{timestamp}"

    # Record the active model and the collection used by the target graph in
    # the same experiment metadata so runs can be compared by both settings.
    metadata = {
        "run_label": run_label,
        "started_at_utc": started_at.isoformat(),
        "dataset_name": EVAL_DATASET_NAME,
        "target": "rag_only_target_html_recursive",
        "model": model_name,
        "retrieval_collection": collection_name,
        "retrieval_n_results": 10,
        "chunking": "HTML headers + recursive chunks; chunk_size=1200; overlap=200",
        "sample_size": sample_size,
        "sample_fraction": sample_fraction,
        "seed": seed,
        "judge": f"{model_name} copy with max_tokens=2048 and reasoning excluded",
    }

    # Submit the recursive HTML-chunk RAG target for evaluation.
    evaluate_target(
        client,
        rag_target,
        experiment_prefix,
        correctness_judge,
        data,
        metadata,
    )


def main() -> None:
    """Parse sampling arguments and run the RAG evaluation."""

    # Define the sampling options in a standard argparse interface.
    parser = argparse.ArgumentParser(
        description="Evaluate the configured RAG target."
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
    parser.add_argument(
        "--run-label",
        help="Short label stored in LangSmith metadata, such as 'baseline' or 'reranker-v1'.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation when evaluating the full dataset.",
    )

    # Parse the user's optional sampling controls.
    args = parser.parse_args()

    # Validate values so downstream sampling logic receives valid inputs.
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be at least 1")
    if args.sample_fraction is not None and not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction must be greater than 0 and at most 1")

    # Previously, omitting sampling arguments silently selected the entire
    # dataset. Ask for confirmation interactively so an accidental full run is
    # visible, while --yes keeps the script usable in automation.
    if (
        args.sample_size is None
        and args.sample_fraction is None
        and not args.yes
    ):
        answer = input(
            "No sample argument supplied; evaluate the full dataset? [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("Evaluation cancelled.")
            return

    # Prompt for a descriptive label when running interactively. This label
    # makes it easy to identify which code/configuration produced each run.
    run_label = args.run_label
    if run_label is None and not args.yes:
        run_label = input(
            "Run label [manual]: "
        ).strip() or "manual"
    if run_label is None:
        run_label = "manual"

    # Construct the client after argument validation; it reads credentials
    # from the process environment.
    client = Client()
    evaluate_rag_target(
        client,
        rag_target=target,
        rag_model=model,
        collection_name=CHROMA_HTML_COLLECTION_NAME,
        sample_size=args.sample_size,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        run_label=run_label,
    )


if __name__ == "__main__":
    # Run the CLI only when this module is executed directly.
    main()
