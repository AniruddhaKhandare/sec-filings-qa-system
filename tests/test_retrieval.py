import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sec_qa_system import VectorStore, HybridRetriever

class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore component"""
    
    def setUp(self):
        self.vector_store = VectorStore(persist_directory="test_chroma_db")
    
    def test_vector_store_initialization(self):
        """Test vector store initialization"""
        self.assertIsNotNone(self.vector_store.client)
        self.assertIsNotNone(self.vector_store.collection)
        self.assertIsNotNone(self.vector_store.encoder)
    
    def tearDown(self):
        """Clean up test artifacts"""
        import shutil
        try:
            shutil.rmtree("test_chroma_db")
        except:
            pass

class TestHybridRetriever(unittest.TestCase):
    """Test cases for HybridRetriever component"""
    
    def setUp(self):
        self.vector_store = VectorStore(persist_directory="test_chroma_db")
        self.retriever = HybridRetriever(self.vector_store)
    
    def test_retriever_initialization(self):
        """Test retriever initialization"""
        self.assertIsNotNone(self.retriever.vector_store)
        self.assertIsNotNone(self.retriever.tfidf_vectorizer)
        self.assertFalse(self.retriever.tfidf_fitted)

if __name__ == '__main__':
    unittest.main()
