#!/usr/bin/env python3
"""
Benchmark Tests for SEC Filings QA System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sec_qa_system import SECQASystem, SystemBenchmark
import time

def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite"""
    qa_system = SECQASystem()
    benchmark = SystemBenchmark(qa_system)
    
    print("🚀 Running Comprehensive Benchmark Suite")
    print("=" * 50)
    
    results = benchmark.run_evaluation_suite()
    
    print("\n📊 Benchmark Results:")
    print(f"Success Rate: {results['evaluation_summary']['success_rate']:.1f}%")
    print(f"Average Response Time: {results['performance_metrics']['avg_response_time_seconds']:.2f}s")
    print(f"Average Sources: {results['performance_metrics']['avg_sources_per_answer']:.1f}")
    
    return results

if __name__ == "__main__":
    run_comprehensive_benchmark()
