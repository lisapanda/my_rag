#with chroma_v4, all-MiniLM-L6-v2(default) + larger chunk size + excluding sources 
import getpass
import os
import uuid
from typing import List

import chromadb 
from config import CHROMA_HTML_COLLECTION_NAME, DATA_DIR
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
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


retrieval_prompt = PromptTemplate(
    template="""You are a teacher grading a quiz. You will be given: 
    1/ a QUESTION
    2/ A FACT provided by the student

    You are grading RELEVANCE RECALL:
    A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
    A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
    1 is the highest (best) score. 0 is the lowest score you can give. 

    Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset.

    Question: {question} \n
    Fact: \n\n {documents} \n\n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """,
    input_variables=["question", "documents"],
)
retrieval_grader = retrieval_prompt | model | JsonOutputParser()


generation_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 

    Use the following documents to answer the question. 

    If you don't know the answer, just say that you don't know. 

    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Documents: {documents} 
    Answer: 
    """,
    input_variables=["question", "documents"],
)
rag_chain = generation_prompt | model | StrOutputParser()

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
    generation = rag_chain.invoke({"documents": documents, "question": question})
    steps = state["steps"]
    steps.append("generate_answer")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
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


def web_search(state):
    """Add web results to the context."""
    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")

    web_results = web_search_tool.invoke({"query": question})
    documents.extend(
        [
            Document(page_content=result["content"], metadata={"url": result["url"]})
            for result in web_results
        ]
    )
    return {"documents": documents, "question": question, "steps": steps}


def decide_to_generate(state):
    """Decide whether to search again or generate an answer."""
    search = state["search"]
    if search == "Yes":
        return "search"
    return "generate"


def build_custom_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    # workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
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
    workflow.add_edge("generate", END)

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
