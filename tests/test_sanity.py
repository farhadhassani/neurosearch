import sys
import os
import unittest

# Add src to path so we can import neurosearch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestNeurosearchImports(unittest.TestCase):
    def test_imports(self):
        """Test that all modules can be imported."""
        try:
            import neurosearch
            from neurosearch.data import esci_loader, semantic_id_builder, indexing_pipeline
            from neurosearch.retrieval import dense_retriever, generative_retriever, hybrid_fusion
            from neurosearch.rag import query_expander, context_builder, rag_ranker
            from neurosearch.eval import metrics, experiments
        except ImportError as e:
            self.fail(f"Import failed: {e}")

    def test_instantiation(self):
        """Test that classes can be instantiated."""
        from neurosearch.data.esci_loader import ESCILoader
        from neurosearch.data.semantic_id_builder import SemanticIDBuilder
        # from neurosearch.data.indexing_pipeline import IndexingPipeline # Requires dependencies
        # from neurosearch.retrieval.dense_retriever import DenseRetriever # Requires sentence-transformers
        # from neurosearch.retrieval.generative_retriever import GenerativeRetriever # Requires transformers
        from neurosearch.rag.query_expander import QueryExpander
        # from neurosearch.rag.context_builder import ContextBuilder # Requires retrievers
        from neurosearch.rag.rag_ranker import RAGRanker

        loader = ESCILoader()
        self.assertIsNotNone(loader)
        
        builder = SemanticIDBuilder()
        self.assertIsNotNone(builder)

        expander = QueryExpander()
        self.assertIsNotNone(expander)

        ranker = RAGRanker()
        self.assertIsNotNone(ranker)

if __name__ == '__main__':
    unittest.main()
