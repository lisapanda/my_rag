"""Create the LangSmith evaluation dataset from the ground-truth CSV.

Run this module directly when the CSV-backed evaluation dataset needs to be
created or populated in LangSmith.
"""

import sys
from pathlib import Path

import pandas as pd
from langsmith import Client

# Resolve the repository root so this script can import project configuration
# regardless of the directory from which it is launched.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import EVAL_DATASET_NAME, GROUND_TRUTH_CSV


def create_dataset(client: Client) -> None:
    """Load ground truth and create the corresponding LangSmith examples.

    Each CSV row becomes one dataset example with an input question, a
    reference answer, and source metadata used to identify the originating
    article and location.
    """

    # Convert each CSV row into a dictionary so fields can be mapped to the
    # input, reference output, and metadata structures expected by LangSmith.
    examples = pd.read_csv(GROUND_TRUTH_CSV).to_dict(orient="records")

    # Reuse an existing dataset when present; otherwise create the named
    # dataset that will contain the uploaded evaluation examples.
    if not client.has_dataset(dataset_name=EVAL_DATASET_NAME):
        dataset = client.create_dataset(EVAL_DATASET_NAME)
    else:
        dataset = client.read_dataset(dataset_name=EVAL_DATASET_NAME)

    # LangSmith separates the model input from the reference output. The
    # target receives the question, while the evaluator compares its answer
    # with the ground-truth answer stored here.
    inputs = [{"input": example["question"]} for example in examples]
    outputs = [{"output": example["answer"]} for example in examples]

    # Preserve provenance so an evaluation result can be traced back to the
    # source document, article, section, and page in the ground-truth data.
    metadata = [
        {
            "source_file": example["source_file"],
            "article": example["article_title"],
            "section": example["section"],
            "page": example["page"],
        }
        for example in examples
    ]

    # Upload all examples in one request and associate them with the selected
    # dataset by its server-generated ID.
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_id=dataset.id,
    )

    # Report the upload count so the user can verify the dataset was populated.
    print(f"Uploaded {len(examples)} examples to dataset '{EVAL_DATASET_NAME}'")


def main() -> None:
    """Create a LangSmith client and upload the ground-truth dataset."""

    # The client reads LangSmith authentication settings from the environment.
    client = Client()
    create_dataset(client)


if __name__ == "__main__":
    # Keep imports safe for reuse while allowing direct CLI execution.
    main()
