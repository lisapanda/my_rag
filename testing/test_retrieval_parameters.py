"""Exercise parameter plumbing without initializing models or a database."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import re
import unittest
from unittest.mock import Mock
import uuid

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    # Execute the real functions in isolation: module imports create live clients.
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ParameterTests(unittest.TestCase):
    def test_graph_targets_forward_default_and_override(self):
        for name in ("custom_graph_retrieval_rerank", "custom_graph_no_web_search_html_recursive_chunks"):
            query = Mock(return_value=["context"])
            graph = Mock()
            ns = load_functions(f"rag_design/{name}.py", {"target", "predict_custom_agent_local_answer", "retrieve"},
                                {"uuid": uuid, "custom_graph": graph, "_movie_titles": [],
                                 "collection": object(), "query_movie_documents": query})
            def invoke(state, config):
                ns["retrieve"](state)
                return {"generation": "answer", "steps": state["steps"]}
            graph.invoke.side_effect = invoke
            for function in ("target", "predict_custom_agent_local_answer"):
                ns[function]({"input": "question"})
                self.assertEqual(query.call_args.kwargs["n_results"], 10)
                ns[function]({"input": "question"}, retrieval_n_results=23)
                self.assertEqual(query.call_args.kwargs["n_results"], 23)
                with self.assertRaises(ValueError):
                    ns[function]({"input": "question"}, retrieval_n_results=0)

    def test_evaluation_metadata_matches_target_limit(self):
        submit = Mock()
        target = Mock(return_value={"response": "answer"})
        ns = load_functions("testing/evaluate_target.py", {"evaluate_rag_target"}, {
            "Client": object, "BaseChatModel": object, "cast": lambda _, value: value,
            "create_correctness_evaluator": Mock(), "select_examples": Mock(return_value=[]),
            "datetime": datetime, "timezone": timezone, "re": re,
            "EVAL_DATASET_NAME": "test", "evaluate_target": submit,
        })
        for kwargs, expected in [({}, 10), ({"retrieval_n_results": 23}, 23)]:
            ns["evaluate_rag_target"](Mock(), target, Mock(), "test", **kwargs)
            args = submit.call_args.args
            args[1]({"input": "question"})
            self.assertEqual(target.call_args.kwargs["retrieval_n_results"], expected)
            self.assertEqual(args[5]["retrieval_n_results"], expected)
        with self.assertRaises(ValueError):
            ns["evaluate_rag_target"](Mock(), target, Mock(), "test", retrieval_n_results=0)


if __name__ == "__main__":
    unittest.main()
