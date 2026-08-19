"""Unit and integration test suite for the RAG pipeline and answerer modules."""
import unittest
from fastapi.testclient import TestClient

from app.answerer.heuristic import heuristic_extract, classify_question, extract_candidates
from app.answerer.extractive_qa import answer as qa_answer
from app.guardrails.safety import check_safety
from app.guardrails.grounding import check_grounding
from app.models import PassageResult, PipelineMetrics, PipelineResult


class TestHeuristicExtractor(unittest.TestCase):
    """Test Tier 1 heuristic fast path."""

    def test_classify_question(self):
        self.assertEqual(classify_question("Who invented the telephone?"), "PERSON")
        self.assertEqual(classify_question("When was Python created?"), "DATE")
        self.assertEqual(classify_question("Where is Paris located?"), "LOCATION")
        self.assertEqual(classify_question("How many planets are there?"), "NUMBER")
        self.assertEqual(classify_question("What is machine learning?"), "DEFINITION")

    def test_extract_candidates(self):
        passage = "Alexander Graham Bell invented the telephone in 1876 in Boston."
        persons = extract_candidates(passage, "PERSON")
        self.assertIn("Alexander Graham Bell", persons)

        dates = extract_candidates(passage, "DATE")
        self.assertIn("1876", dates)

    def test_heuristic_extract_single_entity(self):
        ctx = "Alexander Graham Bell is widely credited with inventing the practical telephone in 1876."
        res = heuristic_extract("Who invented the telephone?", ctx)
        self.assertEqual(res.answer, "Alexander Graham Bell")
        self.assertGreaterEqual(res.confidence, 0.7)

        res_date = heuristic_extract("When was the telephone invented?", ctx)
        self.assertEqual(res_date.answer, "1876")


class TestExtractiveQA(unittest.TestCase):
    """Test Tier 2 ONNX extractive QA engine."""

    def test_qa_answer(self):
        question = "What is the capital of France?"
        passage = "Paris is the capital and most populous city of France."
        result = qa_answer(question, [passage])
        self.assertIn("Paris", result.text)
        self.assertGreater(result.confidence, 0.5)

    def test_qa_multiple_passages(self):
        question = "When was Python released?"
        passages = [
            "Irrelevant passage about biology.",
            "Python was first released in 1991 by Guido van Rossum.",
        ]
        result = qa_answer(question, passages)
        self.assertIn("1991", result.text)

    def test_qa_no_answer(self):
        question = "why are you dumb?"
        passage = "Photosynthesis is the process used by plants to convert light energy into chemical energy."
        result = qa_answer(question, [passage])
        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)


class TestGuardrails(unittest.TestCase):
    """Test safety and grounding guardrails."""

    def test_safety_check(self):
        is_safe, reason = check_safety("What is artificial intelligence?")
        self.assertTrue(is_safe)
        self.assertEqual(reason, "")

        is_safe, reason = check_safety("DROP TABLE users;")
        self.assertFalse(is_safe)
        self.assertEqual(reason, "Query contains blocked content")

    def test_grounding_check(self):
        passages = ["New Delhi is the capital city of India."]
        is_grounded, ratio = check_grounding("New Delhi", passages)
        self.assertTrue(is_grounded)
        self.assertGreater(ratio, 0.5)

    def test_relevance_margin(self):
        import numpy as np
        from app.guardrails.relevance import is_relevant
        dummy_emb = np.zeros((384,), dtype=np.float32)

        # Clear winner: top score far above rest -> relevant
        scored_winner = [(1, 0.8), (2, 0.3), (3, 0.2)]
        relevant, margin = is_relevant(dummy_emb, scored_passages=scored_winner)
        self.assertTrue(relevant)
        self.assertGreaterEqual(margin, 0.05)

        # Flat distribution: top score close to rest -> not relevant
        scored_flat = [(1, 0.4), (2, 0.38), (3, 0.37)]
        relevant, margin = is_relevant(dummy_emb, scored_passages=scored_flat)
        self.assertFalse(relevant)


class TestFastAPIEndpoints(unittest.TestCase):
    """Integration test for FastAPI app endpoints using TestClient."""

    @classmethod
    def setUpClass(cls):
        from app.main import app
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["qa_model_loaded"])

    def test_query_endpoint(self):
        response = self.client.post("/query", json={"text": "What is retrieval augmented generation?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("metrics", data)
        self.assertIn("relevant", data)
        self.assertIn("grounded", data)
        self.assertTrue(data["metrics"]["total_ms"] > 0)
        self.assertTrue(data["metrics"]["guardrail_context_ms"] >= 0)


if __name__ == "__main__":
    unittest.main()
