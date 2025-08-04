
import os
import re
import json
import sqlite3
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for document chunks"""
    ticker: str
    filing_type: str
    filing_date: str
    section: str
    chunk_id: str
    page_number: Optional[int] = None
    
@dataclass
class QueryContext:
    """Parsed query context"""
    tickers: List[str]
    time_periods: List[str]
    filing_types: List[str]
    query_type: str  # ticker-based, temporal, multi-dimensional
    original_query: str

class SECDataFetcher:
    """Handles fetching and caching SEC filings"""
    
    def __init__(self, cache_dir: str = "sec_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SEC-QA-System research@example.com'
        })
        
    def get_company_filings(self, ticker: str, filing_types: List[str] = None, 
                          start_date: str = None, limit: int = 50) -> List[Dict]:
        """Fetch company filings from SEC EDGAR"""
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K', 'DEF 14A']
            
        # Cache key for this request
        cache_key = f"{ticker}_{'-'.join(filing_types)}_{start_date}_{limit}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            logger.info(f"Loading cached filings for {ticker}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        logger.info(f"Fetching filings for {ticker} from SEC EDGAR")
        
        # SEC EDGAR CIK lookup and filing fetch
        cik = self._get_cik_for_ticker(ticker)
        if not cik:
            logger.warning(f"Could not find CIK for ticker {ticker}")
            return []
            
        filings = self._fetch_filings_from_edgar(cik, filing_types, start_date, limit)
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(filings, f, indent=2)
            
        return filings
    
    def _get_cik_for_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker symbol"""
        try:
            # Use SEC company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(url)
            response.raise_for_status()
            
            companies = response.json()
            for company_data in companies.values():
                if company_data.get('ticker', '').upper() == ticker.upper():
                    cik = str(company_data['cik_str']).zfill(10)
                    return cik
                    
        except Exception as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
            
        return None
    
    def _fetch_filings_from_edgar(self, cik: str, filing_types: List[str], 
                                start_date: str, limit: int) -> List[Dict]:
        """Fetch filings from SEC EDGAR for a specific CIK"""
        try:
            # SEC EDGAR submissions endpoint
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            filings = data.get('filings', {}).get('recent', {})
            
            results = []
            for i in range(len(filings.get('form', []))):
                form_type = filings['form'][i]
                if form_type in filing_types:
                    filing_date = filings['filingDate'][i]
                    accession_number = filings['accessionNumber'][i]
                    
                    # Filter by date if specified
                    if start_date and filing_date < start_date:
                        continue
                        
                    filing_info = {
                        'ticker': data.get('tickers', [''])[0] if data.get('tickers') else '',
                        'filing_type': form_type,
                        'filing_date': filing_date,
                        'accession_number': accession_number,
                        'primary_document': filings['primaryDocument'][i],
                        'cik': cik,
                        'company_name': data.get('name', '')
                    }
                    results.append(filing_info)
                    
                    if len(results) >= limit:
                        break
                        
            return sorted(results, key=lambda x: x['filing_date'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error fetching filings for CIK {cik}: {e}")
            return []
    
    def download_filing_content(self, filing_info: Dict) -> Optional[str]:
        """Download the actual filing content"""
        try:
            accession = filing_info['accession_number'].replace('-', '')
            cik = filing_info['cik']
            primary_doc = filing_info['primary_document']
            
            # Construct EDGAR URL
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
            
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error downloading filing content: {e}")
            return None

class DocumentProcessor:
    """Processes and chunks SEC documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def process_filing(self, content: str, filing_info: Dict) -> List[Tuple[str, DocumentMetadata]]:
        """Process a filing into chunks with metadata"""
        if not content:
            return []
            
        # Parse HTML content
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract text and identify sections
        sections = self._extract_sections(soup)
        
        chunks = []
        for section_name, section_text in sections.items():
            if len(section_text.strip()) < 50:  # Skip very short sections
                continue
                
            section_chunks = self._chunk_text(section_text)
            
            for i, chunk in enumerate(section_chunks):
                metadata = DocumentMetadata(
                    ticker=filing_info['ticker'],
                    filing_type=filing_info['filing_type'],
                    filing_date=filing_info['filing_date'],
                    section=section_name,
                    chunk_id=f"{filing_info['accession_number']}_{section_name}_{i}"
                )
                chunks.append((chunk, metadata))
                
        return chunks
    
    def _extract_sections(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract sections from SEC filing HTML"""
        sections = {}
        
        # Common section patterns in SEC filings
        section_patterns = {
            'Business': r'item\s*1\s*[\.\-]?\s*business',
            'Risk Factors': r'item\s*1a\s*[\.\-]?\s*risk\s*factors',
            'Properties': r'item\s*2\s*[\.\-]?\s*properties',
            'Legal Proceedings': r'item\s*3\s*[\.\-]?\s*legal\s*proceedings',
            'Financial Statements': r'item\s*8\s*[\.\-]?\s*financial\s*statements',
            'MD&A': r'item\s*7\s*[\.\-]?\s*management.?s\s*discussion',
            'Controls': r'item\s*9a\s*[\.\-]?\s*controls\s*and\s*procedures',
        }
        
        text = soup.get_text()
        
        # Try to identify sections using patterns
        for section_name, pattern in section_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                start_pos = matches[0].start()
                # Find next section or end of document
                end_pos = len(text)
                for other_pattern in section_patterns.values():
                    if other_pattern != pattern:
                        next_matches = re.search(other_pattern, text[start_pos + 100:], re.IGNORECASE)
                        if next_matches:
                            candidate_end = start_pos + 100 + next_matches.start()
                            if candidate_end < end_pos:
                                end_pos = candidate_end
                
                section_text = text[start_pos:end_pos]
                sections[section_name] = self._clean_text(section_text)
        
        # If no sections found, treat entire document as one section
        if not sections:
            sections['Full Document'] = self._clean_text(text)
            
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into smaller pieces"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks

class QueryProcessor:
    """Processes and analyzes user queries"""
    
    def __init__(self):
        # Common ticker symbols
        self.known_tickers = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc.'
        }
        
        # Filing type mappings
        self.filing_types = {
            '10-k': '10-K',
            '10k': '10-K',
            'annual': '10-K',
            '10-q': '10-Q',
            '10q': '10-Q',
            'quarterly': '10-Q',
            '8-k': '8-K',
            '8k': '8-K',
            'proxy': 'DEF 14A',
            'def 14a': 'DEF 14A'
        }
    
    def parse_query(self, query: str) -> QueryContext:
        """Parse user query to extract context"""
        query_lower = query.lower()
        
        # Extract tickers
        tickers = self._extract_tickers(query)
        
        # Extract time periods
        time_periods = self._extract_time_periods(query)
        
        # Extract filing types
        filing_types = self._extract_filing_types(query)
        
        # Determine query type
        query_type = self._determine_query_type(tickers, time_periods, filing_types)
        
        return QueryContext(
            tickers=tickers,
            time_periods=time_periods,
            filing_types=filing_types,
            query_type=query_type,
            original_query=query
        )
    
    def _extract_tickers(self, query: str) -> List[str]:
        """Extract ticker symbols from query"""
        tickers = []
        query_upper = query.upper()
        
        # Look for explicit ticker mentions
        for ticker, company in self.known_tickers.items():
            if ticker in query_upper or company.lower() in query.lower():
                tickers.append(ticker)
        
        # Look for ticker patterns (3-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        found_tickers = re.findall(ticker_pattern, query.upper())
        for ticker in found_tickers:
            if ticker in self.known_tickers and ticker not in tickers:
                tickers.append(ticker)
                
        return tickers
    
    def _extract_time_periods(self, query: str) -> List[str]:
        """Extract time periods from query"""
        time_periods = []
        
        # Year patterns
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, query)
        time_periods.extend(years)
        
        # Quarter patterns
        quarter_pattern = r'\b(Q[1-4]\s*20\d{2}|20\d{2}\s*Q[1-4])\b'
        quarters = re.findall(quarter_pattern, query, re.IGNORECASE)
        time_periods.extend(quarters)
        
        # Relative time periods
        if any(word in query.lower() for word in ['recent', 'latest', 'current']):
            time_periods.append('recent')
        if any(word in query.lower() for word in ['last year', 'previous year']):
            time_periods.append('last_year')
            
        return time_periods
    
    def _extract_filing_types(self, query: str) -> List[str]:
        """Extract filing types from query"""
        filing_types = []
        query_lower = query.lower()
        
        for pattern, filing_type in self.filing_types.items():
            if pattern in query_lower and filing_type not in filing_types:
                filing_types.append(filing_type)
                
        return filing_types
    
    def _determine_query_type(self, tickers: List[str], time_periods: List[str], 
                            filing_types: List[str]) -> str:
        """Determine the type of query"""
        if len(tickers) == 1 and not time_periods and not filing_types:
            return 'ticker-based'
        elif time_periods and not tickers:
            return 'temporal'
        elif len(tickers) >= 1 and time_periods and filing_types:
            return 'multi-dimensional'
        elif len(tickers) > 1:
            return 'comparative'
        else:
            return 'general'

class VectorStore:
    """Handles vector storage and retrieval using ChromaDB"""
    
    def __init__(self, persist_directory: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="sec_filings",
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_documents(self, chunks: List[Tuple[str, DocumentMetadata]]):
        """Add document chunks to vector store"""
        texts = []
        metadatas = []
        ids = []
        
        for text, metadata in chunks:
            texts.append(text)
            metadatas.append(asdict(metadata))
            ids.append(metadata.chunk_id)
        
        # Generate embeddings
        embeddings = self.encoder.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, filters: Dict = None, n_results: int = 10) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.encoder.encode([query]).tolist()[0]
        
        # Build where clause for filtering
        where_clause = {}
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    where_clause[key] = {"$in": value}
                else:
                    where_clause[key] = value
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
            
        return formatted_results

class HybridRetriever:
    """Combines semantic and keyword-based retrieval"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.tfidf_fitted = False
        self.documents = []
        self.document_metadata = []
    
    def fit_tfidf(self, documents: List[str], metadata: List[Dict]):
        """Fit TF-IDF vectorizer on documents"""
        self.documents = documents
        self.document_metadata = metadata
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.tfidf_fitted = True
        logger.info("TF-IDF vectorizer fitted")
    
    def search(self, query: str, query_context: QueryContext, n_results: int = 10) -> List[Dict]:
        """Perform hybrid search combining semantic and keyword retrieval"""
        # Build filters based on query context
        filters = {}
        if query_context.tickers:
            filters['ticker'] = query_context.tickers
        if query_context.filing_types:
            filters['filing_type'] = query_context.filing_types
        
        # Semantic search
        semantic_results = self.vector_store.search(query, filters, n_results * 2)
        
        # Keyword search (if TF-IDF is fitted)
        keyword_results = []
        if self.tfidf_fitted:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top keyword matches
            top_indices = similarities.argsort()[-n_results * 2:][::-1]
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold for relevance
                    keyword_results.append({
                        'text': self.documents[idx],
                        'metadata': self.document_metadata[idx],
                        'tfidf_score': similarities[idx]
                    })
        
        # Combine and rank results
        combined_results = self._combine_results(semantic_results, keyword_results, n_results)
        
        return combined_results
    
    def _combine_results(self, semantic_results: List[Dict], keyword_results: List[Dict], 
                        n_results: int) -> List[Dict]:
        """Combine semantic and keyword results with ranking"""
        seen_chunks = set()
        combined = []
        
        # Add semantic results with higher weight
        for result in semantic_results:
            chunk_id = result['metadata']['chunk_id']
            if chunk_id not in seen_chunks:
                result['combined_score'] = (1 - result['distance']) * 0.7  # Semantic weight
                combined.append(result)
                seen_chunks.add(chunk_id)
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result['metadata']['chunk_id']
            if chunk_id not in seen_chunks:
                result['combined_score'] = result['tfidf_score'] * 0.3  # Keyword weight
                combined.append(result)
                seen_chunks.add(chunk_id)
            else:
                # Boost score if found in both
                for existing in combined:
                    if existing['metadata']['chunk_id'] == chunk_id:
                        existing['combined_score'] += result['tfidf_score'] * 0.2
                        break
        
        # Sort by combined score and return top results
        combined.sort(key=lambda x: x['combined_score'], reverse=True)
        return combined[:n_results]

class AnswerGenerator:
    """Generates answers using retrieved context"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        # Note: In production, set OPENAI_API_KEY environment variable
        
    def generate_answer(self, query: str, context_chunks: List[Dict], 
                       query_context: QueryContext) -> Dict[str, Any]:
        """Generate answer using retrieved context"""
        # Prepare context
        context_text = self._prepare_context(context_chunks)
        
        # Create prompt
        prompt = self._create_prompt(query, context_text, query_context)
        
        try:
            # Generate answer using OpenAI (mock implementation)
            answer = self._mock_llm_response(query, context_chunks, query_context)
            
            # Extract sources
            sources = self._extract_sources(context_chunks)
            
            return {
                'answer': answer,
                'sources': sources,
                'context_chunks_used': len(context_chunks),
                'query_type': query_context.query_type,
                'confidence': self._estimate_confidence(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': "I apologize, but I encountered an error generating the answer.",
                'sources': [],
                'error': str(e)
            }
    
    def _prepare_context(self, context_chunks: List[Dict]) -> str:
        """Prepare context text from retrieved chunks"""
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            metadata = chunk['metadata']
            text = chunk['text']
            
            context_part = f"""
Source {i+1}: {metadata['ticker']} - {metadata['filing_type']} - {metadata['filing_date']} - {metadata['section']}
{text}
---
            """
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, query_context: QueryContext) -> str:
        """Create prompt for answer generation"""
        return f"""
You are a financial research analyst tasked with answering questions about SEC filings. 
Use the provided context to answer the user's question accurately and comprehensively.

Query: {query}
Query Type: {query_context.query_type}
Relevant Tickers: {', '.join(query_context.tickers) if query_context.tickers else 'N/A'}
Time Periods: {', '.join(query_context.time_periods) if query_context.time_periods else 'N/A'}

Context from SEC Filings:
{context}

Instructions:
1. Answer the question comprehensively using the provided context
2. Always cite your sources using the format [Source X]
3. If information is not available in the context, clearly state this
4. For comparative questions, structure your answer clearly
5. Highlight any uncertainty or limitations in the available data
6. Focus on factual information from the filings

Answer:
        """
    
    def _mock_llm_response(self, query: str, context_chunks: List[Dict], 
                          query_context: QueryContext) -> str:
        """Mock LLM response for demonstration (replace with actual OpenAI call)"""
        # This is a simplified mock response
        # In production, this would call OpenAI's API
        
        tickers = query_context.tickers
        query_lower = query.lower()
        
        if 'revenue' in query_lower and 'compare' in query_lower:
            return f"""Based on the SEC filings analysis for {', '.join(tickers)}, here are the key revenue insights:

**Revenue Comparison:**
The companies show different revenue growth patterns based on their business models and market positions. [Source 1, Source 2]

**Key Findings:**
- Technology companies demonstrate strong recurring revenue streams from subscription services
- Revenue diversity varies significantly across the analyzed companies
- Recent quarters show resilience despite market challenges [Source 3]

**Limitations:** The analysis is based on available filing data and may not reflect the most recent performance metrics."""

        elif 'risk' in query_lower:
            return f"""Risk factor analysis for {', '.join(tickers) if tickers else 'the analyzed companies'}:

**Primary Risk Categories:**
1. **Market and Competition Risks**: Intense competition and market volatility [Source 1]
2. **Regulatory Risks**: Changing regulatory environment and compliance costs [Source 2]
3. **Technology Risks**: Cybersecurity threats and technology disruption [Source 3]

**Company-Specific Considerations:**
Each company emphasizes different risk priorities based on their industry exposure and business model.

**Note:** This analysis is based on the most recent available SEC filings in our database."""

        else:
            return f"""Based on the SEC filings analysis, here are the key findings for your query about {query}:

The available filing data provides insights into the requested information. Key points include:

- Analysis covers multiple filing periods and document types [Source 1]
- Information is drawn from official SEC filings including 10-K, 10-Q, and 8-K reports [Source 2]
- Data reflects the companies' own disclosures and representations [Source 3]

For more specific insights, please provide additional details about the particular aspects you'd like to explore.

**Data Limitations:** Analysis is based on available filing data and standard SEC reporting requirements."""
    
    def _extract_sources(self, context_chunks: List[Dict]) -> List[Dict]:
        """Extract source information from context chunks"""
        sources = []
        for i, chunk in enumerate(context_chunks):
            metadata = chunk['metadata']
            sources.append({
                'source_id': i + 1,
                'ticker': metadata['ticker'],
                'filing_type': metadata['filing_type'],
                'filing_date': metadata['filing_date'],
                'section': metadata['section'],
                'chunk_id': metadata['chunk_id']
            })
        return sources
    
    def _estimate_confidence(self, context_chunks: List[Dict]) -> float:
        """Estimate confidence based on context quality"""
        if not context_chunks:
            return 0.0
        
        # Simple confidence estimation based on:
        # - Number of relevant chunks
        # - Diversity of sources
        # - Recency of data
        
        num_chunks = len(context_chunks)
        unique_sources = len(set(chunk['metadata']['ticker'] for chunk in context_chunks))
        
        base_confidence = min(0.9, 0.3 + (num_chunks * 0.1))
        diversity_bonus = min(0.1, unique_sources * 0.02)
        
        return base_confidence + diversity_bonus

class SECQASystem:
    """Main SEC QA System orchestrating all components"""
    
    def __init__(self, data_dir: str = "sec_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.fetcher = SECDataFetcher()
        self.processor = DocumentProcessor()
        self.query_processor = QueryProcessor()
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.answer_generator = AnswerGenerator()
        
        # Database for tracking processed documents
        self.db_path = self.data_dir / "processed_docs.db"
        self._init_database()
        
        logger.info("SEC QA System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_filings (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                filing_type TEXT,
                filing_date TEXT,
                accession_number TEXT UNIQUE,
                processed_date TEXT,
                chunk_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def ingest_company_data(self, tickers: List[str], filing_types: List[str] = None,
                          start_date: str = None, limit_per_company: int = 20):
        """Ingest SEC data for specified companies"""
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K', 'DEF 14A']
        
        all_chunks = []
        all_texts = []
        all_metadata = []
        
        for ticker in tickers:
            logger.info(f"Processing data for {ticker}")
            
            # Fetch filings
            filings = self.fetcher.get_company_filings(
                ticker, filing_types, start_date, limit_per_company
            )
            
            for filing in filings:
                # Check if already processed
                if self._is_filing_processed(filing['accession_number']):
                    logger.info(f"Skipping already processed filing: {filing['accession_number']}")
                    continue
                
                # Download content
                content = self.fetcher.download_filing_content(filing)
                if not content:
                    continue
                
                # Process document
                chunks = self.processor.process_filing(content, filing)
                
                if chunks:
                    all_chunks.extend(chunks)
                    
                    # Prepare for TF-IDF fitting
                    for chunk_text, metadata in chunks:
                        all_texts.append(chunk_text)
                        all_metadata.append(asdict(metadata))
                    
                    # Mark as processed
                    self._mark_filing_processed(filing, len(chunks))
                    logger.info(f"Processed {len(chunks)} chunks from {filing['filing_type']} filing")
        
        if all_chunks:
            # Add to vector store
            self.vector_store.add_documents(all_chunks)
            
            # Fit TF-IDF for hybrid retrieval
            self.retriever.fit_tfidf(all_texts, all_metadata)
            
            logger.info(f"Ingested {len(all_chunks)} total chunks from {len(tickers)} companies")
    
    def _is_filing_processed(self, accession_number: str) -> bool:
        """Check if filing is already processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM processed_filings WHERE accession_number = ?",
            (accession_number,)
        )
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def _mark_filing_processed(self, filing: Dict, chunk_count: int):
        """Mark filing as processed in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO processed_filings 
            (ticker, filing_type, filing_date, accession_number, processed_date, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            filing['ticker'],
            filing['filing_type'],
            filing['filing_date'],
            filing['accession_number'],
            datetime.now().isoformat(),
            chunk_count
        ))
        
        conn.commit()
        conn.close()
    
    def query(self, question: str, n_results: int = 10) -> Dict[str, Any]:
        """Process a query and return answer with sources"""
        logger.info(f"Processing query: {question}")
        
        # Parse query
        query_context = self.query_processor.parse_query(question)
        logger.info(f"Query type: {query_context.query_type}")
        logger.info(f"Extracted tickers: {query_context.tickers}")
        
        # Retrieve relevant documents
        relevant_chunks = self.retriever.search(question, query_context, n_results)
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Generate answer
        result = self.answer_generator.generate_answer(question, relevant_chunks, query_context)
        
        # Add query metadata
        result['query_context'] = asdict(query_context)
        result['retrieval_stats'] = {
            'chunks_retrieved': len(relevant_chunks),
            'unique_companies': len(set(chunk['metadata']['ticker'] for chunk in relevant_chunks)),
            'filing_types_covered': list(set(chunk['metadata']['filing_type'] for chunk in relevant_chunks))
        }
        
        return result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get filing counts by ticker
        cursor.execute('''
            SELECT ticker, filing_type, COUNT(*) as count, SUM(chunk_count) as total_chunks
            FROM processed_filings 
            GROUP BY ticker, filing_type
        ''')
        
        filing_stats = cursor.fetchall()
        
        # Get total stats
        cursor.execute('SELECT COUNT(*) FROM processed_filings')
        total_filings = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(chunk_count) FROM processed_filings')
        total_chunks = cursor.fetchone()[0] or 0
        
        conn.close()
        
        # Get vector store stats
        try:
            vector_count = self.vector_store.collection.count()
        except:
            vector_count = 0
        
        return {
            'total_filings_processed': total_filings,
            'total_chunks_created': total_chunks,
            'vector_store_count': vector_count,
            'filings_by_company': [
                {
                    'ticker': row[0],
                    'filing_type': row[1],
                    'count': row[2],
                    'chunks': row[3]
                }
                for row in filing_stats
            ]
        }

def main():
    """Example usage and testing"""
    
    # Sample companies for testing
    test_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("=" * 60)
    print("SEC Filings QA System - Quantitative Researcher Challenge")
    print("=" * 60)
    
    # Initialize system
    qa_system = SECQASystem()
    
    # Check if we need to ingest data
    stats = qa_system.get_system_stats()
    print(f"\nCurrent System Stats:")
    print(f"- Total filings processed: {stats['total_filings_processed']}")
    print(f"- Total chunks created: {stats['total_chunks_created']}")
    print(f"- Vector store count: {stats['vector_store_count']}")
    
    if stats['total_filings_processed'] == 0:
        print("\nNo data found. Ingesting sample company data...")
        print("Note: This is a demonstration with limited data fetching.")
        
        # In a real implementation, this would fetch actual SEC data
        print("Simulating data ingestion for companies:", test_companies)
        
        # Mock some processed filings for demonstration
        import sqlite3
        conn = sqlite3.connect(qa_system.db_path)
        cursor = conn.cursor()
        
        mock_filings = [
            ('AAPL', '10-K', '2023-10-27', '0000320193-23-000106', '2024-01-15T10:00:00', 45),
            ('AAPL', '10-Q', '2023-08-03', '0000320193-23-000077', '2024-01-15T10:00:00', 32),
            ('MSFT', '10-K', '2023-07-27', '0000789019-23-000076', '2024-01-15T10:00:00', 52),
            ('MSFT', '10-Q', '2023-10-25', '0000789019-23-000103', '2024-01-15T10:00:00', 38),
            ('GOOGL', '10-K', '2023-02-02', '0001652044-23-000016', '2024-01-15T10:00:00', 67),
        ]
        
        for filing in mock_filings:
            cursor.execute('''
                INSERT OR REPLACE INTO processed_filings 
                (ticker, filing_type, filing_date, accession_number, processed_date, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', filing)
        
        conn.commit()
        conn.close()
        
        print("Mock data ingestion completed.")
    
    # Sample evaluation questions from the challenge
    sample_questions = [
        "What are the primary revenue drivers for major technology companies, and how have they evolved?",
        "Compare R&D spending trends across companies. What insights about innovation investment strategies?",
        "What are the most commonly cited risk factors across industries?",
        "How do companies describe climate-related risks? Notable industry differences?",
        "How are companies positioning regarding AI and automation? Strategic approaches?",
        "What are Apple's main risk factors according to their latest 10-K?",
        "Compare Apple and Microsoft revenues from their most recent filings",
        "How has Apple's revenue guidance changed over time?",
        "What significant working capital changes occurred for technology companies?",
        "Analyze recent executive compensation changes across the technology sector"
    ]
    
    print(f"\n{'='*60}")
    print("SAMPLE QUERY DEMONSTRATIONS")
    print("="*60)
    
    # Demonstrate system with sample queries
    for i, question in enumerate(sample_questions[:3]):  # Limit to first 3 for demo
        print(f"\n🔍 Query {i+1}: {question}")
        print("-" * 50)
        
        try:
            result = qa_system.query(question)
            
            print(f"Query Type: {result['query_context']['query_type']}")
            print(f"Identified Tickers: {result['query_context']['tickers']}")
            print(f"Confidence: {result.get('confidence', 'N/A'):.2f}" if isinstance(result.get('confidence'), (int, float)) else f"Confidence: {result.get('confidence', 'N/A')}")
            
            print(f"\n📋 Answer:")
            print(result['answer'])
            
            print(f"\n📚 Sources ({len(result['sources'])} total):")
            for source in result['sources'][:3]:  # Show first 3 sources
                print(f"  - {source['ticker']} {source['filing_type']} ({source['filing_date']}) - {source['section']}")
            
            if len(result['sources']) > 3:
                print(f"  ... and {len(result['sources']) - 3} more sources")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.error(f"Query error: {e}")
    
    # Interactive mode
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter your questions about SEC filings (type 'quit' to exit, 'stats' for system info)")
    
    while True:
        try:
            user_query = input("\n🤔 Your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            elif user_query.lower() == 'stats':
                stats = qa_system.get_system_stats()
                print("\n📊 System Statistics:")
                print(f"  Total filings: {stats['total_filings_processed']}")
                print(f"  Total chunks: {stats['total_chunks_created']}")
                print(f"  Companies covered: {len(set(f['ticker'] for f in stats['filings_by_company']))}")
                continue
            elif not user_query:
                continue
            
            print("\n🔄 Processing your query...")
            result = qa_system.query(user_query)
            
            print(f"\n📋 Answer:")
            print(result['answer'])
            
            print(f"\n📚 Sources:")
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"  {i}. {source['ticker']} - {source['filing_type']} ({source['filing_date']})")
            
            retrieval_stats = result.get('retrieval_stats', {})
            print(f"\n📈 Retrieval Stats:")
            print(f"  Chunks analyzed: {retrieval_stats.get('chunks_retrieved', 0)}")
            print(f"  Companies covered: {retrieval_stats.get('unique_companies', 0)}")
            print(f"  Filing types: {', '.join(retrieval_stats.get('filing_types_covered', []))}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Interactive query error: {e}")

class SystemBenchmark:
    """Benchmark and evaluation utilities"""
    
    def __init__(self, qa_system: SECQASystem):
        self.qa_system = qa_system
    
    def run_evaluation_suite(self) -> Dict[str, Any]:
        """Run the complete evaluation suite from the challenge"""
        
        evaluation_questions = [
            {
                'id': 1,
                'question': "What are the primary revenue drivers for major technology companies, and how have they evolved?",
                'category': 'comparative',
                'expected_elements': ['revenue streams', 'evolution', 'multiple companies']
            },
            {
                'id': 2,
                'question': "Compare R&D spending trends across companies. What insights about innovation investment strategies?",
                'category': 'comparative',
                'expected_elements': ['R&D spending', 'trends', 'innovation strategy']
            },
            {
                'id': 3,
                'question': "Identify significant working capital changes for financial services companies and driving factors.",
                'category': 'sector-specific',
                'expected_elements': ['working capital', 'changes', 'driving factors']
            },
            {
                'id': 4,
                'question': "What are the most commonly cited risk factors across industries? How do same-sector companies prioritize differently?",
                'category': 'risk-analysis',
                'expected_elements': ['risk factors', 'industry comparison', 'prioritization']
            },
            {
                'id': 5,
                'question': "How do companies describe climate-related risks? Notable industry differences?",
                'category': 'thematic',
                'expected_elements': ['climate risks', 'industry differences', 'descriptions']
            }
        ]
        
        results = []
        for question_data in evaluation_questions:
            print(f"\n🔍 Evaluating Question {question_data['id']}: {question_data['question']}")
            
            start_time = datetime.now()
            try:
                result = self.qa_system.query(question_data['question'])
                end_time = datetime.now()
                
                evaluation = {
                    'question_id': question_data['id'],
                    'category': question_data['category'],
                    'response_time_seconds': (end_time - start_time).total_seconds(),
                    'sources_count': len(result.get('sources', [])),
                    'confidence': result.get('confidence', 0),
                    'query_type': result.get('query_context', {}).get('query_type', 'unknown'),
                    'chunks_retrieved': result.get('retrieval_stats', {}).get('chunks_retrieved', 0),
                    'companies_covered': result.get('retrieval_stats', {}).get('unique_companies', 0),
                    'answer_length': len(result.get('answer', '')),
                    'has_sources': len(result.get('sources', [])) > 0,
                    'addresses_expected_elements': self._check_expected_elements(
                        result.get('answer', ''), question_data['expected_elements']
                    )
                }
                
                results.append(evaluation)
                print(f"  ✅ Completed in {evaluation['response_time_seconds']:.2f}s")
                print(f"  📊 {evaluation['sources_count']} sources, {evaluation['companies_covered']} companies")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'question_id': question_data['id'],
                    'error': str(e),
                    'response_time_seconds': (datetime.now() - start_time).total_seconds()
                })
        
        return self._compile_evaluation_report(results)
    
    def _check_expected_elements(self, answer: str, expected_elements: List[str]) -> Dict[str, bool]:
        """Check if answer addresses expected elements"""
        answer_lower = answer.lower()
        element_check = {}
        
        for element in expected_elements:
            # Simple keyword matching - could be improved with more sophisticated NLP
            element_words = element.lower().split()
            element_check[element] = any(word in answer_lower for word in element_words)
        
        return element_check
    
    def _compile_evaluation_report(self, results: List[Dict]) -> Dict[str, Any]:
        """Compile evaluation results into a comprehensive report"""
        successful_results = [r for r in results if 'error' not in r]
        error_count = len(results) - len(successful_results)
        
        if not successful_results:
            return {
                'overall_success_rate': 0.0,
                'total_errors': error_count,
                'results': results
            }
        
        # Calculate metrics
        avg_response_time = np.mean([r['response_time_seconds'] for r in successful_results])
        avg_sources = np.mean([r['sources_count'] for r in successful_results])
        avg_confidence = np.mean([r['confidence'] for r in successful_results if isinstance(r['confidence'], (int, float))])
        
        # Query type distribution
        query_types = [r['query_type'] for r in successful_results]
        query_type_dist = {qt: query_types.count(qt) for qt in set(query_types)}
        
        return {
            'evaluation_summary': {
                'total_questions': len(results),
                'successful_responses': len(successful_results),
                'error_count': error_count,
                'success_rate': len(successful_results) / len(results) * 100
            },
            'performance_metrics': {
                'avg_response_time_seconds': round(avg_response_time, 2),
                'avg_sources_per_answer': round(avg_sources, 1),
                'avg_confidence_score': round(avg_confidence, 2) if not np.isnan(avg_confidence) else None,
                'query_type_distribution': query_type_dist
            },
            'detailed_results': results,
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate improvement recommendations based on results"""
        recommendations = []
        
        successful_results = [r for r in results if 'error' not in r]
        if not successful_results:
            recommendations.append("System failed on all test queries - needs fundamental debugging")
            return recommendations
        
        # Check response times
        slow_queries = [r for r in successful_results if r['response_time_seconds'] > 5]
        if slow_queries:
            recommendations.append(f"Consider optimizing retrieval performance - {len(slow_queries)} queries took >5 seconds")
        
        # Check source coverage
        low_source_queries = [r for r in successful_results if r['sources_count'] < 3]
        if low_source_queries:
            recommendations.append(f"Improve source retrieval - {len(low_source_queries)} queries returned <3 sources")
        
        # Check company coverage
        low_company_coverage = [r for r in successful_results if r['companies_covered'] < 2]
        if low_company_coverage and len(successful_results) > 0:
            recommendations.append("Consider ingesting data from more companies for better coverage")
        
        return recommendations

if __name__ == "__main__":
    main()
