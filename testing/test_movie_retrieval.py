"""Focused retrieval checks: no model calls or persistent database writes."""
import unittest
from rag_design.movie_retrieval import query_movie_documents


class Collection:
    def __init__(self):
        self.rows = [("a", "8½", "A director faces a creative crisis."),
                     ("b", "8½", "The production budget was $230,000."),
                     ("c", "The Matrix", "Neo enters the Matrix.")]
        self.queries = []

    def get(self, include, where=None):
        rows = [r for r in self.rows if not where or r[1] == where["movie_title"]]
        return {"ids": [r[0] for r in rows], "documents": [r[2] for r in rows]}

    def query(self, query_texts, n_results, include, where=None):
        self.queries.append(where)
        rows = self.get(include, where)
        # Deliberately put the exact lexical match below the semantic match.
        return {"ids": [rows["ids"][:n_results]],
                "documents": [rows["documents"][:n_results]]}


class RetrievalTests(unittest.TestCase):
    def test_sparse_finds_exact_term_without_dense_call(self):
        c = Collection()
        result = query_movie_documents(c, "budget", [], 1, mode="sparse")
        self.assertIn("$230,000", result[0])
        self.assertEqual(c.queries, [])

    def test_hybrid_combines_and_respects_title(self):
        c = Collection()
        result = query_movie_documents(c, "8½ budget", ["8½", "The Matrix"], 2)
        self.assertIn("$230,000", result[0])
        self.assertEqual(len(set(result)), 2)
        self.assertEqual(c.queries, [{"movie_title": "8½"}])

    def test_empty_title_filter_falls_back(self):
        result = query_movie_documents(Collection(), "Missing budget", ["Missing"], mode="sparse")
        self.assertIn("$230,000", result[0])

    def test_empty_corpus_and_no_keyword_match(self):
        c = Collection()
        self.assertEqual(query_movie_documents(c, "xyzxyz", [], mode="sparse"), [])
        c.rows = []
        self.assertEqual(query_movie_documents(c, "budget", []), [])

    def test_dense_and_validation(self):
        c = Collection()
        self.assertIn("creative crisis", query_movie_documents(c, "budget", [], 1, mode="dense")[0])
        with self.assertRaises(ValueError):
            query_movie_documents(c, "budget", [], mode="other")
        with self.assertRaises(ValueError):
            query_movie_documents(c, "budget", [], n_results=0)


if __name__ == "__main__":
    unittest.main()
