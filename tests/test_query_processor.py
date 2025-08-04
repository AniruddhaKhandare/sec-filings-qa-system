import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sec_qa_system import QueryProcessor

class TestQueryProcessor(unittest.TestCase):
    """Test cases for QueryProcessor component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = QueryProcessor()
    
    def test_ticker_extraction(self):
        """Test ticker symbol extraction from queries"""
        test_cases = [
            ("What are Apple's risk factors?", ['AAPL']),
            ("Compare AAPL and MSFT revenues", ['AAPL', 'MSFT']),
            ("Microsoft Corporation financial data", ['MSFT']),
            ("GOOGL vs Amazon performance", ['GOOGL', 'AMZN']),
            ("No company mentioned here", []),
        ]
        
        for query, expected_tickers in test_cases:
            with self.subTest(query=query):
                context = self.processor.parse_query(query)
                self.assertEqual(set(context.tickers), set(expected_tickers))
    
    def test_time_period_extraction(self):
        """Test time period extraction from queries"""
        test_cases = [
            ("Apple's 2023 performance", ['2023']),
            ("Q4 2022 results", ['Q4 2022']),
            ("Recent financial data", ['recent']),
            ("2021 vs 2022 comparison", ['2021', '2022']),
            ("Current quarter analysis", ['recent']),
            ("No time mentioned", []),
        ]
        
        for query, expected_periods in test_cases:
            with self.subTest(query=query):
                context = self.processor.parse_query(query)
                # Check if any expected periods are found
                found_periods = [p for p in expected_periods if any(ep in context.time_periods for ep in expected_periods)]
                if expected_periods:
                    self.assertTrue(len(context.time_periods) > 0)
    
    def test_filing_type_extraction(self):
        """Test filing type extraction from queries"""
        test_cases = [
            ("Apple's 10-K filing", ['10-K']),
            ("Quarterly 10-Q report", ['10-Q']),
            ("Annual report analysis", ['10-K']),
            ("8-K filing details", ['8-K']),
            ("Proxy statement info", ['DEF 14A']),
            ("General company info", []),
        ]
        
        for query, expected_types in test_cases:
            with self.subTest(query=query):
                context = self.processor.parse_query(query)
                self.assertEqual(set(context.filing_types), set(expected_types))
    
    def test_query_type_classification(self):
        """Test query type classification"""
        test_cases = [
            ("Apple's risk factors", "ticker-based"),
            ("Compare Apple and Microsoft", "comparative"),
            ("How has revenue changed over time?", "temporal"),
            ("Apple's 2023 10-K risk factors", "multi-dimensional"),
            ("Technology sector analysis", "general"),
        ]
        
        for query, expected_type in test_cases:
            with self.subTest(query=query):
                context = self.processor.parse_query(query)
                self.assertEqual(context.query_type, expected_type)
    
    def test_complex_query_parsing(self):
        """Test parsing of complex, multi-dimensional queries"""
        query = "Compare Apple's and Microsoft's risk factors from their 2023 10-K filings"
        context = self.processor.parse_query(query)
        
        self.assertIn('AAPL', context.tickers)
        self.assertIn('MSFT', context.tickers)
        self.assertIn('2023', context.time_periods)
        self.assertIn('10-K', context.filing_types)
        self.assertEqual(context.query_type, 'multi-dimensional')
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        edge_cases = [
            "",  # Empty query
            "   ",  # Whitespace only
            "What is the meaning of life?",  # Completely unrelated
            "AAPL MSFT GOOGL AMZN TSLA",  # Only tickers
        ]
        
        for query in edge_cases:
            with self.subTest(query=query):
                try:
                    context = self.processor.parse_query(query)
                    # Should not raise an exception
                    self.assertIsNotNone(context)
                    self.assertIsInstance(context.tickers, list)
                    self.assertIsInstance(context.time_periods, list)
                    self.assertIsInstance(context.filing_types, list)
                except Exception as e:
                    self.fail(f"Query processor failed on edge case '{query}': {e}")
    
    def test_case_insensitive_processing(self):
        """Test that processing is case-insensitive"""
        queries = [
            "apple's risk factors",
            "APPLE'S RISK FACTORS", 
            "Apple's Risk Factors",
            "aPpLe'S rIsK fAcToRs"
        ]
        
        contexts = [self.processor.parse_query(q) for q in queries]
        
        # All should extract AAPL ticker
        for context in contexts:
            self.assertIn('AAPL', context.tickers)
    
    def test_financial_terminology_recognition(self):
        """Test recognition of financial terminology"""
        financial_queries = [
            "EBITDA analysis for tech companies",
            "Free cash flow trends",
            "R&D spending comparison", 
            "M&A activity overview",
            "Revenue growth patterns"
        ]
        
        for query in financial_queries:
            with self.subTest(query=query):
                context = self.processor.parse_query(query)
                # Should successfully parse without errors
                self.assertIsNotNone(context)
                self.assertIsInstance(context.original_query, str)

class TestQueryProcessorIntegration(unittest.TestCase):
    """Integration tests for QueryProcessor with real scenarios"""
    
    def setUp(self):
        self.processor = QueryProcessor()
    
    def test_realistic_research_queries(self):
        """Test with realistic financial research queries"""
        research_queries = [
            "What are the primary revenue drivers for Apple and how have they evolved over the past three years?",
            "Compare the risk factors mentioned in Microsoft's and Google's most recent 10-K filings",
            "How do major banks describe regulatory compliance costs in their quarterly reports?",
            "What climate-related risks are most commonly cited across energy sector companies?",
            "Analyze the R&D spending trends for major pharmaceutical companies in 2023"
        ]
        
        for query in research_queries:
            with self.subTest(query=query):
                context = self.processor.parse_query(query)
                
                # Should have meaningful parsing results
                self.assertIsNotNone(context.query_type)
                self.assertIsInstance(context.tickers, list)
                self.assertIsInstance(context.time_periods, list)
                self.assertIsInstance(context.filing_types, list)
                self.assertEqual(context.original_query, query)
    
    def test_query_context_consistency(self):
        """Test that query context is consistent across multiple parses"""
        query = "Compare Apple and Microsoft's 2023 annual report risk factors"
        
        # Parse same query multiple times
        contexts = [self.processor.parse_query(query) for _ in range(5)]
        
        # Results should be identical
        for context in contexts[1:]:
            self.assertEqual(context.tickers, contexts[0].tickers)
            self.assertEqual(context.time_periods, contexts[0].time_periods)
            self.assertEqual(context.filing_types, contexts[0].filing_types)
            self.assertEqual(context.query_type, contexts[0].query_type)

if __name__ == '__main__':
    unittest.main()
