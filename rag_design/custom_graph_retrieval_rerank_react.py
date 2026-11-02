#with chroma_v4, all-MiniLM-L6-v2(default) + larger chunk size + excluding sources 
import getpass
import os
import sys
import uuid
from pathlib import Path
from typing import List

# Make project imports work when this file is launched directly from any folder.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb 
from config import CHROMA_HTML_COLLECTION_NAME, DATA_DIR
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from rag_design.movie_prompts import retrieval_prompt, generation_prompt, answer_check_prompt
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

try:
    from .models import model
    from .movie_retrieval import query_movie_documents
except ImportError:  # pragma: no cover - fallback for direct script execution
    from models import model
    from movie_retrieval import query_movie_documents


def _set_env_if_missing(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


# _set_env_if_missing("TAVILY_API_KEY")

client = chromadb.PersistentClient(path=str(DATA_DIR))
collection = client.get_or_create_collection(CHROMA_HTML_COLLECTION_NAME)
_movie_titles = None

# Design-specific settings reported by the evaluation runner.
EVALUATION_METADATA = {
    "retrieval_mode": "hybrid",
    "chunking": "HTML headers + recursive chunks; chunk_size=1200; overlap=200",
    "answer_review": True,
    "max_generation_attempts": 2,
}


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]
    retrieval_n_results: NotRequired[int]
    attempts: NotRequired[int]
    answer_satisfied: NotRequired[bool]
    feedback: NotRequired[str]


# Compose the imported templates with the model and output parsers.
retrieval_grader = retrieval_prompt | model | JsonOutputParser()
rag_chain = generation_prompt | model | StrOutputParser()
answer_checker = answer_check_prompt | model | JsonOutputParser()

# web_search_tool = TavilySearchResults(k=3)


def retrieve(state):
    """Retrieve documents."""
    global _movie_titles
    question = state["question"]
    if _movie_titles is None:
        # Read once per process; older collections without titles stay unfiltered.
        metadata = collection.get(include=["metadatas"])["metadatas"] or []
        _movie_titles = sorted({item["movie_title"] for item in metadata
                                if item and item.get("movie_title")})
    documents = query_movie_documents(
        collection, question, _movie_titles,
        n_results=state.get("retrieval_n_results", 10),
    )

    steps = state["steps"]
    steps.append("retrieve_documents")
    return {"documents": documents, "question": question, "steps": steps}


def generate(state):
    """Generate answer."""
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({
        "documents": documents, "question": question,
        "feedback": state.get("feedback", "No previous review."),
    })
    steps = state["steps"]
    steps.append("generate_answer")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "attempts": state.get("attempts", 0) + 1,
        "steps": steps,
    }


def check_answer(state):
    """Review the generated answer and record feedback for a possible retry."""
    review = answer_checker.invoke({
        "question": state["question"], "documents": state["documents"],
        "generation": state["generation"],
    })
    # Do not use bool(value): bool("false") would incorrectly approve the answer.
    if not isinstance(review, dict) or not isinstance(review.get("satisfied"), bool):
        raise ValueError("Answer checker must return a boolean 'satisfied' field")
    return {
        "answer_satisfied": review["satisfied"],
        "feedback": str(review.get("feedback", "")),
        "steps": state["steps"] + ["check_answer"],
    }


def route_after_check(state):
    """Finish on approval, otherwise allow one revision before abstaining."""
    if state["answer_satisfied"]:
        return "end"
    if state.get("attempts", 0) < 2:
        return "retry"
    return "abstain"


def abstain(state):
    """Avoid returning a draft that failed the final evidence check."""
    return {
        "generation": "I couldn't produce a complete, supported answer from the retrieved documents.",
        "steps": state["steps"] + ["abstain"],
    }


def grade_documents(state):
    """Filter retrieved documents based on relevance."""
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"

    for document in documents:
        score = retrieval_grader.invoke({"question": question, "documents": document})
        grade = score.get("score")
        if grade == "yes":
            filtered_docs.append(document)
        else:
            search = "Yes"

    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }


# def web_search(state):
#     """Add web results to the context."""
#     question = state["question"]
#     documents = state.get("documents", [])
#     steps = state["steps"]
#     steps.append("web_search")

#     web_results = web_search_tool.invoke({"query": question})
#     documents.extend(
#         [
#             Document(page_content=result["content"], metadata={"url": result["url"]})
#             for result in web_results
#         ]
#     )
#     return {"documents": documents, "question": question, "steps": steps}


# def decide_to_generate(state):
#     """Decide whether to search again or generate an answer."""
#     search = state["search"]
#     if search == "Yes":
#         return "search"
#     return "generate"


def build_custom_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    # workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("check_answer", check_answer)
    workflow.add_node("abstain", abstain)
    # workflow.add_node("web_search", web_search)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    # workflow.add_conditional_edges(
    #     "grade_documents",
    #     decide_to_generate,
    #     {
    #         "search": "web_search",
    #         "generate": "generate",
    #     },
    # )
    # workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "check_answer")
    workflow.add_conditional_edges(
        "check_answer", route_after_check,
        {"end": END, "retry": "generate", "abstain": "abstain"},
    )
    workflow.add_edge("abstain", END)

    return workflow.compile()


custom_graph = build_custom_graph()


def predict_custom_agent_local_answer(example: dict, retrieval_n_results: int = 10):
    if retrieval_n_results < 1:
        raise ValueError("retrieval_n_results must be positive")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_dict = custom_graph.invoke(
        {"question": example["input"], "steps": [],
         "retrieval_n_results": retrieval_n_results},
        config,
    )
    return {"response": state_dict["generation"], "steps": state_dict["steps"]}


def target(example: dict, retrieval_n_results: int = 10) -> dict:
    if retrieval_n_results < 1:
        raise ValueError("retrieval_n_results must be positive")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_dict = custom_graph.invoke(
        {"question": example["input"], "steps": [],
         "retrieval_n_results": retrieval_n_results},
        config,
    )
    return {"response": state_dict["generation"]}


def main() -> None:
    """Print the compiled graph as Mermaid text without running a question."""
    # print(custom_graph.get_graph().draw_mermaid())
    print(custom_graph.get_graph().draw_ascii())

if __name__ == "__main__":
    main()
