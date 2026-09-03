from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
GROUND_TRUTH_CSV = DATA_DIR / "movie_wikipedia_ground_truth.csv"

CHROMA_COLLECTION_NAME = "my_movies"
CHROMA_HTML_COLLECTION_NAME = "my_movies_html"
CHROMA_HTML_REGULAR_COLLECTION_NAME = "my_movies_html_regular"

EVAL_DATASET_NAME = "Corrective RAG Agent Movie Testing_v2"
EVAL_MAX_CONCURRENCY = 4
