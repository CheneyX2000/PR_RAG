# tests/test_rag_quality.py
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

class TestRAGQuality:
    @pytest.mark.asyncio
    async def test_retrieval_relevance(self, rag_system):
        query = "What is PgVector?"
        documents = await rag_system.retrieve(query, top_k=5)
        
        # Test retrieval quality
        assert len(documents) > 0
        assert all(doc.similarity_score > 0.7 for doc in documents)
    
    @pytest.mark.asyncio
    async def test_generation_quality(self, rag_system):
        test_case = LLMTestCase(
            input="Explain PgVector indexing",
            actual_output=await rag_system.generate("Explain PgVector indexing"),
            context=["PgVector supports HNSW and IVFFlat indexes..."]
        )
        
        relevancy_metric = AnswerRelevancyMetric(threshold=0.8)
        faithfulness_metric = FaithfulnessMetric(threshold=0.9)
        
        assert_test(test_case, [relevancy_metric, faithfulness_metric])