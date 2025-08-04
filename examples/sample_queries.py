#!/usr/bin/env python3
"""
Sample Queries for SEC Filings QA System

This file demonstrates various types of queries that the system can handle,
organized by category and complexity level.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sec_qa_system import SECQASystem
import time

def run_sample_queries():
    """Run a comprehensive set of sample queries to demonstrate system capabilities"""
    
    print("🔍 SEC Filings QA System - Sample Query Demonstrations")
    print("=" * 70)
    
    # Initialize system
    qa_system = SECQASystem()
    
    # Query categories with examples
    query_categories = {
        "Ticker-Based Queries": [
            "What are Apple's main risk factors?",
            "Describe Microsoft's business segments",
            "What is Google's revenue breakdown by product?",
            "How does Amazon describe its competitive advantages?",
            "What are Tesla's manufacturing capabilities?"
        ],
        
        "Comparative Queries": [
            "Compare Apple and Microsoft's revenue growth strategies",
            "How do Google and Meta approach AI development differently?",
            "Compare R&D spending between tech companies",
            "What are the key differences in risk factors between Amazon and Walmart?",
            "How do JPMorgan and Bank of America describe regulatory risks?"
        ],
        
        "Temporal Queries": [
            "How has Apple's revenue guidance changed over time?",
            "What trends emerge in Microsoft's quarterly reports?",
            "How have tech companies' AI investments evolved?",
            "What changes in risk factors occurred after 2020?",
            "How has climate risk disclosure evolved across industries?"
        ],
        
        "Multi-Dimensional Queries": [
            "What did Apple report about supply chain risks in their 2023 10-K?",
            "Compare Q3 2023 earnings guidance across major tech companies",
            "How did financial services companies describe regulatory changes in recent 8-K filings?",
            "What cybersecurity investments were mentioned in 2023 annual reports?",
            "Which companies reported material M&A activity in recent proxy statements?"
        ],
        
        "Financial Analysis Queries": [
            "What are the primary revenue drivers for major technology companies?",
            "How do companies describe working capital management strategies?",
            "What trends appear in executive compensation across industries?",
            "How do companies explain margin pressure in their MD&A sections?",
            "What debt management strategies are most commonly described?"
        ],
        
        "Risk and Compliance Queries": [
            "What are the most commonly cited risk factors across industries?",
            "How do companies describe climate-related risks?",
            "What cybersecurity risks are most frequently mentioned?",
            "How do companies address regulatory compliance costs?",
            "What litigation risks appear most significant across sectors?"
        ],
        
        "Strategic and Innovation Queries": [
            "How are companies positioning regarding artificial intelligence?",
            "What digital transformation initiatives are most common?",
            "How do companies describe their competitive moats?",
            "What innovation investments are prioritized across sectors?",
            "How do companies approach sustainability initiatives?"
        ],
        
        "Market and Industry Queries": [
            "How do companies describe current market conditions?",
            "What supply chain challenges are most frequently cited?",
            "How do companies position against industry disruption?",
            "What geographic expansion strategies are most common?",
            "How do companies describe customer acquisition strategies?"
        ]
    }
    
    # Run queries by category
    for category, queries in query_categories.items():
        print(f"\n📂 {category}")
        print("-" * len(category))
        
        for i, query in enumerate(queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            
            try:
                start_time = time.time()
                result = qa_system.query(query)
                end_time = time.time()
                
                # Display results
                print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds")
                print(f"🎯 Query Type: {result['query_context']['query_type']}")
                
                if result['query_context']['tickers']:
                    print(f"🏢 Companies: {', '.join(result['query_context']['tickers'])}")
                
                if result['query_context']['time_periods']:
                    print(f"📅 Time Periods: {', '.join(result['query_context']['time_periods'])}")
                
                print(f"📊 Sources: {len(result['sources'])} documents")
                print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
                
                # Show abbreviated answer
                answer = result['answer']
                if len(answer) > 300:
                    answer = answer[:300] + "..."
                print(f"💡 Answer: {answer}")
                
                # Show source breakdown
                if result['sources']:
                    print("📚 Top Sources:")
                    for j, source in enumerate(result['sources'][:3], 1):
                        print(f"   {j}. {source['ticker']} - {source['filing_type']} ({source['filing_date']}) - {source['section']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print()  # Add spacing between queries
        
        print("=" * 70)

def demonstrate_query_types():
    """Demonstrate different query type classifications"""
    
    print("\n🔬 Query Type Classification Demonstration")
    print("=" * 50)
    
    qa_system = SECQASystem()
    
    test_queries = [
        ("Apple's revenue growth", "ticker-based"),
        ("Compare Apple and Microsoft revenues", "comparative"),
        ("How has revenue changed over time?", "temporal"),
        ("Apple's 2023 10-K risk factors", "multi-dimensional"),
        ("Technology sector trends", "general"),
    ]
    
    for query, expected_type in test_queries:
        context = qa_system.query_processor.parse_query(query)
        
        print(f"Query: '{query}'")
        print(f"  Expected Type: {expected_type}")
        print(f"  Detected Type: {context.query_type}")
        print(f"  Tickers: {context.tickers}")
        print(f"  Time Periods: {context.time_periods}")
        print(f"  Filing Types: {context.filing_types}")
        print()

def run_performance_benchmark():
    """Run a performance benchmark on sample queries"""
    
    print("\n⚡ Performance Benchmark")
    print("=" * 30)
    
    qa_system = SECQASystem()
    
    benchmark_queries = [
        "What are Apple's main risk factors?",
        "Compare tech companies' R&D spending",
        "How has AI investment changed over time?",
        "What climate risks do companies report?",
        "Describe competitive advantages in tech sector"
    ]
    
    total_time = 0
    results = []
    
    for i, query in enumerate(benchmark_queries, 1):
        print(f"🔍 Benchmark Query {i}: {query}")
        
        start_time = time.time()
        try:
            result = qa_system.query(query)
            end_time = time.time()
            
            query_time = end_time - start_time
            total_time += query_time
            
            results.append({
                'query': query,
                'time': query_time,
                'sources': len(result['sources']),
                'confidence': result.get('confidence', 0),
                'success': True
            })
            
            print(f"  ✅ Completed in {query_time:.2f}s - {len(result['sources'])} sources")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                'query': query,
                'time': 0,
                'sources': 0,
                'confidence': 0,
                'success': False,
                'error': str(e)
            })
    
    # Performance summary
    successful_queries = [r for r in results if r['success']]
    
    if successful_queries:
        avg_time = total_time / len(successful_queries)
        avg_sources = sum(r['sources'] for r in successful_queries) / len(successful_queries)
        avg_confidence = sum(r['confidence'] for r in successful_queries if isinstance(r['confidence'], (int, float))) / len(successful_queries)
        
        print(f"\n📈 Performance Summary:")
        print(f"  Total Queries: {len(benchmark_queries)}")
        print(f"  Successful: {len(successful_queries)}")
        print(f"  Average Time: {avg_time:.2f} seconds")
        print(f"  Average Sources: {avg_sources:.1f}")
        print(f"  Average Confidence: {avg_confidence:.2f}")
        print(f"  Success Rate: {len(successful_queries)/len(benchmark_queries)*100:.1f}%")

def demonstrate_advanced_features():
    """Demonstrate advanced system features"""
    
    print("\n🚀 Advanced Features Demonstration")
    print("=" * 40)
    
    qa_system = SECQASystem()
    
    # 1. System Statistics
    print("📊 System Statistics:")
    stats = qa_system.get_system_stats()
    print(f"  Total Filings: {stats['total_filings_processed']}")
    print(f"  Total Chunks: {stats['total_chunks_created']}")
    print(f"  Vector Store Count: {stats['vector_store_count']}")
    
    if stats['filings_by_company']:
        print("  Companies Coverage:")
        for filing in stats['filings_by_company'][:5]:  # Show first 5
            print(f"    {filing['ticker']}: {filing['count']} {filing['filing_type']} filings")
    
    # 2. Query Context Analysis
    print(f"\n🔍 Query Context Analysis:")
    test_query = "Compare Apple and Microsoft's 2023 10-K risk factors"
    context = qa_system.query_processor.parse_query(test_query)
    
    print(f"  Query: '{test_query}'")
    print(f"  Type: {context.query_type}")
    print(f"  Companies: {context.tickers}")
    print(f"  Time: {context.time_periods}")
    print(f"  Filing Types: {context.filing_types}")
    
    # 3. Retrieval Analytics (if available)
    print(f"\n📈 Retrieval System Features:")
    print("  ✅ Hybrid semantic + keyword search")
    print("  ✅ Financial domain term boosting")
    print("  ✅ Section-aware relevance scoring")
    print("  ✅ Temporal decay for recent filings")
    print("  ✅ Cross-document coherence analysis")
    print("  ✅ Source diversity optimization")

def interactive_query_session():
    """Run an interactive query session"""
    
    print("\n💬 Interactive Query Session")
    print("=" * 35)
    print("Enter your questions about SEC filings")
    print("Commands: 'help' for tips, 'stats' for system info, 'quit' to exit")
    
    qa_system = SECQASystem()
    
    # Provide query suggestions
    suggestions = [
        "What are Apple's main risk factors?",
        "Compare tech companies' AI investments",
        "How do banks describe regulatory risks?",
        "What climate risks do energy companies report?",
        "Describe competitive advantages in healthcare"
    ]
    
    print(f"\n💡 Suggested queries:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    while True:
        try:
            user_input = input(f"\n🤔 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print(f"\n💡 Query Tips:")
                print("  • Be specific about companies (use ticker symbols)")
                print("  • Include time periods (2023, Q4 2022, recent)")
                print("  • Mention filing types (10-K, 10-Q, annual report)")
                print("  • Use comparison words (compare, versus, differences)")
                print("  • Ask about specific topics (risks, revenue, strategy)")
                continue
            elif user_input.lower() == 'stats':
                stats = qa_system.get_system_stats()
                print(f"\n📊 System Stats:")
                print(f"  Filings: {stats['total_filings_processed']}")
                print(f"  Chunks: {stats['total_chunks_created']}")
                print(f"  Companies: {len(set(f['ticker'] for f in stats['filings_by_company']))}")
                continue
            elif not user_input:
                continue
            
            print(f"\n🔄 Processing query...")
            start_time = time.time()
            
            result = qa_system.query(user_input)
            end_time = time.time()
            
            print(f"\n📋 Answer ({end_time - start_time:.2f}s):")
            print(result['answer'])
            
            print(f"\n📚 Sources ({len(result['sources'])} total):")
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"  {i}. {source['ticker']} - {source['filing_type']} ({source['filing_date']})")
            
            if len(result['sources']) > 5:
                print(f"  ... and {len(result['sources']) - 5} more sources")
            
            # Show query analysis
            context = result['query_context']
            print(f"\n🎯 Query Analysis:")
            print(f"  Type: {context['query_type']}")
            if context['tickers']:
                print(f"  Companies: {', '.join(context['tickers'])}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demonstration function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="SEC QA System Sample Queries")
    parser.add_argument('--mode', choices=['all', 'queries', 'types', 'benchmark', 'features', 'interactive'], 
                       default='all', help='Demonstration mode')
    parser.add_argument('--category', help='Specific query category to run')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        demonstrate_query_types()
        demonstrate_advanced_features()
        run_performance_benchmark()
        run_sample_queries()
        
    elif args.mode == 'queries':
        run_sample_queries()
        
    elif args.mode == 'types':
        demonstrate_query_types()
        
    elif args.mode == 'benchmark':
        run_performance_benchmark()
        
    elif args.mode == 'features':
        demonstrate_advanced_features()
        
    elif args.mode == 'interactive':
        interactive_query_session()
    
    print(f"\n🎉 Sample queries demonstration completed!")
    print(f"See docs/technical_documentation.md for more details.")

if __name__ == "__main__":
    main()
