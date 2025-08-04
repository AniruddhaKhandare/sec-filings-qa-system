# tests/__init__.py

"""
Test suite for SEC Filings QA System.

This package contains all unit and integration tests for the SEC Filings QA System.
The test suite is organized into modules that test specific components of the system.

Modules:
    test_query_processor.py: Tests for the EnhancedQueryProcessor class
    test_retrieval.py: Tests for document retrieval and processing components

To run all tests:
    python -m pytest tests/

For coverage reporting:
    pytest --cov=sec_qa_system tests/
"""

# Import test modules to make them available when package is imported
from . import test_query_processor
from . import test_retrieval

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
