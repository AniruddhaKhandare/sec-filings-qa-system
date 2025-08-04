# tests/__init__.py

"""
Test suite for the SEC Filings QA System.

This package includes all unit and integration tests for validating the system's core components.

Modules:
    - test_query_processor.py: Tests for the EnhancedQueryProcessor class
    - test_retrieval.py: Tests for document embedding, storage, and hybrid search

Usage:
    Run all tests:
        $ pytest tests/

    Run with coverage report:
        $ pytest --cov=sec_qa_system tests/
"""

# Import test modules for test discovery
from . import test_query_processor
from . import test_retrieval

__version__ = "0.1.0"
__author__ = "Aniruddha"
__email__ = "your.email@domain.com"
