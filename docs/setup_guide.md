# Setup Guide - SEC Filings QA System

## Quick Start (5 minutes)

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/sec-filings-qa-system.git
cd sec-filings-qa-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"  # For production
export SEC_DATA_DIR="./sec_data"
export SEC_CACHE_DIR="./sec_cache"
```

### 4. Run the System
```python
python sec_qa_system.py
```

## Detailed Installation

### Prerequisites

- **Python 3.8+**: Required for modern typing and async support
- **Git**: For cloning the repository
- **OpenAI API Key**: For production answer generation (optional for demo)
- **Storage**: 50-100GB recommended for comprehensive data

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4GB | 16GB |
| **Storage** | 10GB | 100GB |
| **CPU** | 2 cores | 8 cores |
| **Network** | 10 Mbps | 100 Mbps |

### Python Environment Setup

#### Option 1: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv sec_qa_env

# Activate environment
# On Windows:
sec_qa_env\Scripts\activate
# On macOS/Linux:
source sec_qa_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Conda Environment
```bash
# Create conda environment
conda create -n sec_qa python=3.9

# Activate environment
conda activate sec_qa

# Install dependencies
pip install -r requirements.txt
```

### Dependency Installation

The system requires several key packages:

#### Core Dependencies
```bash
# Data processing
pip install pandas>=1.5.0 numpy>=1.21.0

# Web scraping and parsing
pip install requests>=2.28.0 beautifulsoup4>=4.11.0 lxml>=4.9.0

# NLP and embeddings
pip install sentence-transformers>=2.2.0 tiktoken>=0.4.0

# Vector database
pip install chromadb>=0.4.0

# Machine learning
pip install scikit-learn>=1.1.0

# LLM integration
pip install openai>=1.0.0
```

#### Optional Dependencies
```bash
# Development tools
pip install pytest>=7.0.0 black>=22.0.0 flake8>=5.0.0

# Visualization (optional)
pip install matplotlib>=3.5.0 seaborn>=0.11.0

# Jupyter notebooks (optional)
pip install jupyter>=1.0.0
```

### Configuration

#### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for production LLM integration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom data directories
SEC_DATA_DIR=./sec_data
SEC_CACHE_DIR=./sec_cache
CHROMA_DB_DIR=./chroma_db

# Optional: Performance tuning
MAX_WORKERS=4
CACHE_SIZE_MB=1024
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Optional: API rate limiting
SEC_API_DELAY=0.1
MAX_RETRIES=3
```

#### Directory Structure Setup

The system will create necessary directories automatically, but you can pre-create them:

```bash
mkdir -p sec_data sec_cache chroma_db logs
```

### Initial Data Ingestion

#### Quick Demo Data
```python
from sec_qa_system import SECQASystem

# Initialize system
qa_system = SECQASystem()

# Ingest data for major tech companies
demo_companies = ['AAPL', 'MSFT', 'GOOGL']
qa_system.ingest_company_data(demo_companies, limit_per_company=5)
```

#### Production Data Ingestion
```python
# Full data ingestion for comprehensive coverage
production_companies = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
    'JPM', 'BAC', 'WFC', 'GS', 'MS',  # Financial
    'JNJ', 'PFE', 'UNH', 'ABBV',     # Healthcare
    'XOM', 'CVX', 'COP',             # Energy
    'WMT', 'HD', 'PG', 'KO'          # Consumer
]

qa_system.ingest_company_data(
    production_companies, 
    filing_types=['10-K', '10-Q', '8-K', 'DEF 14A'],
    start_date='2020-01-01',
    limit_per_company=50
)
```

## Configuration Options

### System Configuration

#### Basic Configuration
```python
# Initialize with custom settings
qa_system = SECQASystem(
    data_dir="./custom_sec_data",
    chunk_size=1500,
    chunk_overlap=300
)
```

#### Advanced Configuration
```python
from sec_qa_system import SECQASystem, DocumentProcessor, VectorStore

# Custom document processor
processor = DocumentProcessor(
    chunk_size=800,
    chunk_overlap=150
)

# Custom vector store
vector_store = VectorStore(
    persist_directory="./custom_chroma",
    embedding_model="all-mpnet-base-v2"
)

# Initialize with custom components
qa_system = SECQASystem(
    data_dir="./sec_data",
    processor=processor,
    vector_store=vector_store
)
```

### Retrieval Configuration

#### Hybrid Retrieval Tuning
```python
# Adjust semantic vs keyword weights
qa_system.retriever.semantic_weight = 0.7
qa_system.retriever.keyword_weight = 0.3

# Financial term boosting
qa_system.retriever.financial_terms.update({
    'artificial intelligence': 2.5,
    'machine learning': 2.3,
    'blockchain': 2.2,
    'cybersecurity': 2.4
})

# Section importance weights
qa_system.retriever.section_weights['risk']['Cybersecurity'] = 3.0
qa_system.retriever.section_weights['strategic']['Innovation'] = 2.8
```

### Answer Generation Configuration

#### OpenAI Integration
```python
from sec_qa_system import AnswerGenerator

# Configure OpenAI parameters
answer_generator = AnswerGenerator(
    model_name="gpt-4",
    temperature=0.1,
    max_tokens=1000
)

qa_system.answer_generator = answer_generator
```

## Troubleshooting

### Common Issues

#### 1. ChromaDB Installation Issues
```bash
# If ChromaDB installation fails
pip install --upgrade pip setuptools wheel
pip install chromadb --no-cache-dir

# Alternative: Use conda-forge
conda install -c conda-forge chromadb
```

#### 2. Sentence Transformers Download Issues
```python
# Pre-download models
from sentence_transformers import SentenceTransformer

# Download required models
SentenceTransformer('all-MiniLM-L6-v2')
SentenceTransformer('all-mpnet-base-v2')  # Optional, higher quality
```

#### 3. Memory Issues
```python
# Reduce chunk size for limited memory
qa_system = SECQASystem()
qa_system.processor.chunk_size = 500
qa_system.processor.chunk_overlap = 100

# Process companies in smaller batches
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
for company in companies:
    qa_system.ingest_company_data([company], limit_per_company=10)
```

#### 4. SEC API Rate Limiting
```python
# Add delays for SEC API compliance
import time

class RateLimitedFetcher(SECDataFetcher):
    def get_company_filings(self, *args, **kwargs):
        time.sleep(0.1)  # 100ms delay between requests
        return super().get_company_filings(*args, **kwargs)

qa_system.fetcher = RateLimitedFetcher()
```

### Performance Optimization

#### 1. Parallel Processing
```python
import multiprocessing

# Use multiple workers for ingestion
qa_system.ingest_company_data(
    companies,
    max_workers=min(4, multiprocessing.cpu_count())
)
```

#### 2. Batch Processing
```python
# Process multiple companies in batches
def batch_ingest(companies, batch_size=5):
    for i in range(0, len(companies), batch_size):
        batch = companies[i:i+batch_size]
        qa_system.ingest_company_data(batch)
        print(f"Processed batch {i//batch_size + 1}")

batch_ingest(production_companies)
```

#### 3. Memory Management
```python
# Clear caches periodically
import gc

def memory_efficient_ingest(companies):
    for i, company in enumerate(companies):
        qa_system.ingest_company_data([company])
        
        if i % 10 == 0:  # Clear memory every 10 companies
            gc.collect()
            print(f"Memory cleared after {i+1} companies")
```

## Testing the Installation

### Basic Functionality Test
```python
# Test script: test_installation.py
from sec_qa_system import SECQASystem

def test_basic_functionality():
    print("Testing SEC QA System installation...")
    
    # Initialize system
    qa_system = SECQASystem()
    print("✅ System initialization successful")
    
    # Test components
    stats = qa_system.get_system_stats()
    print(f"✅ Database connection successful: {stats}")
    
    # Test query processing
    query_context = qa_system.query_processor.parse_query("What are Apple's risk factors?")
    print(f"✅ Query processing successful: {query_context.tickers}")
    
    print("🎉 Installation test completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()
```

### Full System Test
```python
# Test script: test_full_system.py
def test_full_system():
    qa_system = SECQASystem()
    
    # Ingest small amount of test data
    test_companies = ['AAPL']
    qa_system.ingest_company_data(test_companies, limit_per_company=1)
    
    # Test query
    result = qa_system.query("What are Apple's main business segments?")
    
    assert result['answer'], "Answer generation failed"
    assert result['sources'], "Source attribution failed"
    assert result['query_context']['tickers'] == ['AAPL'], "Query parsing failed"
    
    print("✅ Full system test passed!")

if __name__ == "__main__":
    test_full_system()
```

## Next Steps

### 1. Run Sample Queries
```python
# Try these sample queries after setup
sample_queries = [
    "What are Apple's main risk factors?",
    "Compare Microsoft and Google's revenue trends",
    "How do tech companies describe AI investments?",
    "What are the most common cybersecurity risks mentioned?"
]

for query in sample_queries:
    result = qa_system.query(query)
    print(f"Query: {query}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Sources: {len(result['sources'])}")
    print("-" * 50)
```

### 2. Explore Advanced Features
```python
# Benchmark the system
from sec_qa_system import SystemBenchmark

benchmark = SystemBenchmark(qa_system)
results = benchmark.run_evaluation_suite()
print(results['evaluation_summary'])
```

### 3. Customize for Your Use Case
```python
# Add domain-specific terms
qa_system.retriever.financial_terms.update({
    'your_industry_term': 2.5,
    'specific_metric': 2.2
})

# Adjust section weights
qa_system.retriever.section_weights['custom_category'] = {
    'Your_Section': 3.0,
    'Another_Section': 2.5
}
```

## Support and Documentation

- **Technical Documentation**: `docs/technical_documentation.md`
- **Architecture Overview**: `docs/architecture_overview.md`
- **Evaluation Results**: `docs/evaluation_results.md`
- **Example Queries**: `examples/sample_queries.py`
- **Benchmark Tests**: `examples/benchmark_tests.py`

For additional support, check the GitHub issues or create a new issue with:
- Your system configuration
- Error messages or logs
- Steps to reproduce the problem
- Expected vs actual behavior
