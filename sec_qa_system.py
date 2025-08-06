import os
import re
import json
import sqlite3
import requests
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
from sklearn.cluster import KMeans
import spacy
from textstat import flesch_reading_ease

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Enhanced metadata for document chunks"""
    ticker: str
    filing_type: str
    filing_date: str
    section: str
    chunk_id: str
    page_number: Optional[int] = None
    company_name: Optional[str] = None
    cik: Optional[str] = None
    word_count: Optional[int] = None
    readability_score: Optional[float] = None
    
@dataclass
class QueryContext:
    """Enhanced parsed query context"""
    tickers: List[str]
    time_periods: List[str]
    filing_types: List[str]
    query_type: str
    original_query: str
    entities: List[str] = None
    intent_keywords: List[str] = None
    complexity_score: float = 0.0

@dataclass
class PerformanceMetrics:
    """System performance tracking"""
    query_start_time: float
    retrieval_time: float
    generation_time: float
    total_time: float
    chunks_processed: int
    cache_hits: int
    api_calls: int

class EnhancedSECDataFetcher:
    """Complete SEC data fetching implementation with caching and rate limiting"""
    
    def __init__(self, cache_dir: str = "sec_cache", rate_limit: float = 0.1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rate_limit = rate_limit  # seconds between requests
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SEC-QA-System research@quantfirm.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
        
        # Company ticker to CIK mapping cache
        self.ticker_cik_cache = {}
        self._load_ticker_mappings()
        
    def _load_ticker_mappings(self):
        """Load and cache ticker to CIK mappings"""
        cache_file = self.cache_dir / "ticker_cik_mappings.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.ticker_cik_cache = json.load(f)
            logger.info(f"Loaded {len(self.ticker_cik_cache)} ticker mappings from cache")
        else:
            self._fetch_ticker_mappings()
    
    def _fetch_ticker_mappings(self):
        """Fetch complete ticker to CIK mappings from SEC"""
        try:
            self._rate_limit()
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(url)
            response.raise_for_status()
            
            companies = response.json()
            for company_data in companies.values():
                ticker = company_data.get('ticker', '').upper()
                cik = str(company_data['cik_str']).zfill(10)
                self.ticker_cik_cache[ticker] = {
                    'cik': cik,
                    'name': company_data.get('title', ''),
                    'exchange': company_data.get('exchange', '')
                }
            
            # Save to cache
            cache_file = self.cache_dir / "ticker_cik_mappings.json"
            with open(cache_file, 'w') as f:
                json.dump(self.ticker_cik_cache, f, indent=2)
                
            logger.info(f"Fetched and cached {len(self.ticker_cik_cache)} ticker mappings")
            
        except Exception as e:
            logger.error(f"Error fetching ticker mappings: {e}")
    
    def _rate_limit(self):
        """Implement rate limiting for SEC requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def get_company_filings(self, ticker: str, filing_types: List[str] = None, 
                          start_date: str = None, limit: int = 50) -> List[Dict]:
        """Enhanced filing fetching with complete implementation"""
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K', 'DEF 14A', 'FORM 3', 'FORM 4', 'FORM 5']
            
        # Cache key for this request
        cache_key = f"{ticker}_{'-'.join(filing_types)}_{start_date}_{limit}"
        cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = self.cache_dir / f"filings_{cache_key_hash}.json"
        
        if cache_file.exists():
            # Check cache age (refresh if older than 1 day)
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                logger.info(f"Loading cached filings for {ticker}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        logger.info(f"Fetching filings for {ticker} from SEC EDGAR")
        
        # Get CIK for ticker
        company_info = self.ticker_cik_cache.get(ticker.upper())
        if not company_info:
            logger.warning(f"Could not find CIK for ticker {ticker}")
            return []
            
        cik = company_info['cik']
        filings = self._fetch_filings_from_edgar(cik, filing_types, start_date, limit, company_info)
        
        # Cache the results
        try:
            with open(cache_file, 'w') as f:
                json.dump(filings, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not cache filings: {e}")
            
        return filings
    
    def _fetch_filings_from_edgar(self, cik: str, filing_types: List[str], 
                                start_date: str, limit: int, company_info: Dict) -> List[Dict]:
        """Complete implementation of SEC EDGAR filing fetch"""
        try:
            self._rate_limit()
            
            # SEC EDGAR submissions endpoint
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            filings = data.get('filings', {}).get('recent', {})
            
            results = []
            forms = filings.get('form', [])
            
            for i in range(len(forms)):
                form_type = forms[i]
                if form_type in filing_types:
                    filing_date = filings['filingDate'][i]
                    
                    # Filter by date if specified
                    if start_date and filing_date < start_date:
                        continue
                        
                    filing_info = {
                        'ticker': company_info.get('name', '').split()[0] if company_info.get('name') else '',
                        'company_name': company_info.get('name', ''),
                        'filing_type': form_type,
                        'filing_date': filing_date,
                        'accession_number': filings['accessionNumber'][i],
                        'primary_document': filings['primaryDocument'][i],
                        'cik': cik,
                        'report_date': filings.get('reportDate', [None] * len(forms))[i],
                        'accepted_date': filings.get('acceptanceDateTime', [None] * len(forms))[i],
                        'file_number': filings.get('fileNumber', [None] * len(forms))[i],
                        'film_number': filings.get('filmNumber', [None] * len(forms))[i]
                    }
                    results.append(filing_info)
                    
                    if len(results) >= limit:
                        break
            
            # Check for older filings if needed
            if len(results) < limit and 'older' in data.get('filings', {}):
                older_filings = self._fetch_older_filings(cik, filing_types, start_date, 
                                                        limit - len(results), company_info)
                results.extend(older_filings)
                        
            return sorted(results, key=lambda x: x['filing_date'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error fetching filings for CIK {cik}: {e}")
            return []
    
    def _fetch_older_filings(self, cik: str, filing_types: List[str], start_date: str, 
                       remaining_limit: int, company_info: Dict) -> List[Dict]:
    """Fetch older filings from additional archives"""
    try:
        self._rate_limit()
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = self.session.get(url)
        
        data = response.json()
        older_files = data.get('filings', {}).get('files', [])
        
        results = []
        for older_file in older_files:
            if len(results) >= remaining_limit:
                break
                
            try:
                older_url = f"https://data.sec.gov/submissions/{older_file['name']}"
                older_response = self.session.get(older_url)
                older_data = older_response.json()
                
                for i, form_type in enumerate(older_data.get('form', [])):
                    if form_type in filing_types and len(results) < remaining_limit:
                        filing_info = {
                            'ticker': company_info.get('name', '').split()[0],
                            'company_name': company_info.get('name', ''),
                            'filing_type': form_type,
                            'filing_date': older_data['filingDate'][i],
                            'accession_number': older_data['accessionNumber'][i],
                            'primary_document': older_data['primaryDocument'][i],
                            'cik': cik
                        }
                        results.append(filing_info)
            except Exception as file_error:
                logger.warning(f"Error processing older file {older_file.get('name')}: {file_error}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching older filings: {e}")
        return []

        except Exception as e:
            logger.error(f"Error fetching older filings: {e}")
            return []
    
    def download_filing_content(self, filing_info: Dict) -> Optional[str]:
        """Enhanced filing content download with error handling"""
        try:
            accession = filing_info['accession_number'].replace('-', '')
            cik = filing_info['cik']
            primary_doc = filing_info['primary_document']
            
            # Check cache first
            cache_key = f"{accession}_{primary_doc}"
            cache_file = self.cache_dir / f"content_{cache_key}.html"
            
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            
            self._rate_limit()
            
            # Construct EDGAR URL
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            content = response.text
            
            # Cache the content
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                logger.warning(f"Could not cache content: {e}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error downloading filing content: {e}")
            return None
    
    async def download_multiple_filings(self, filing_list: List[Dict], 
                                      max_concurrent: int = 5) -> Dict[str, str]:
        """Asynchronous batch downloading of multiple filings"""
        async def download_single(session, filing):
            try:
                accession = filing['accession_number'].replace('-', '')
                cik = filing['cik']
                primary_doc = filing['primary_document']
                
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
                
                await asyncio.sleep(self.rate_limit)  # Rate limiting
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return filing['accession_number'], content
                    else:
                        logger.warning(f"Failed to download {accession}: {response.status}")
                        return filing['accession_number'], None
                        
            except Exception as e:
                logger.error(f"Error in async download: {e}")
                return filing['accession_number'], None
        
        results = {}
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [download_single(session, filing) for filing in filing_list]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in completed:
                if isinstance(result, tuple):
                    accession, content = result
                    results[accession] = content
        
        return results

class EnhancedDocumentProcessor:
    """Enhanced document processing with NLP and structure analysis"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Enhanced section patterns with regex groups
        self.section_patterns = {
            'Business': r'item\s*1\s*[\.\-]?\s*(business|overview)',
            'Risk Factors': r'item\s*1a\s*[\.\-]?\s*risk\s*factors',
            'Properties': r'item\s*2\s*[\.\-]?\s*properties',
            'Legal Proceedings': r'item\s*3\s*[\.\-]?\s*legal\s*proceedings',
            'Controls and Procedures': r'item\s*9a\s*[\.\-]?\s*controls\s*and\s*procedures',
            'MD&A': r'item\s*7\s*[\.\-]?\s*management.?s\s*discussion\s*and\s*analysis',
            'Financial Statements': r'item\s*8\s*[\.\-]?\s*financial\s*statements',
            'Directors and Officers': r'item\s*10\s*[\.\-]?\s*directors.*officers',
            'Executive Compensation': r'item\s*11\s*[\.\-]?\s*executive\s*compensation',
            'Security Ownership': r'item\s*12\s*[\.\-]?\s*security\s*ownership',
            'Exhibits': r'item\s*15\s*[\.\-]?\s*exhibits',
            'Financial Data': r'consolidated\s*statements?|balance\s*sheet|income\s*statement',
            'Notes to Financial Statements': r'notes?\s*to\s*(the\s*)?consolidated\s*financial\s*statements',
            'Insider Trading': r'form\s*[345]\s*|insider\s*trading',
            'Proxy Statement': r'definitive\s*proxy\s*statement|def\s*14a'
        }
        
    def process_filing(self, content: str, filing_info: Dict) -> List[Tuple[str, DocumentMetadata]]:
        """Enhanced filing processing with structure analysis"""
        if not content:
            return []
            
        # Parse HTML content with enhanced extraction
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract sections with enhanced detection
        sections = self._extract_sections_enhanced(soup, filing_info)
        
        chunks = []
        for section_name, section_data in sections.items():
            section_text = section_data['text']
            if len(section_text.strip()) < 50:  # Skip very short sections
                continue
                
            section_chunks = self._chunk_text_enhanced(section_text, section_data)
            
            for i, (chunk, chunk_metadata) in enumerate(section_chunks):
                metadata = DocumentMetadata(
                    ticker=filing_info.get('ticker', ''),
                    filing_type=filing_info['filing_type'],
                    filing_date=filing_info['filing_date'],
                    section=section_name,
                    chunk_id=f"{filing_info['accession_number']}_{section_name}_{i}",
                    company_name=filing_info.get('company_name', ''),
                    cik=filing_info.get('cik', ''),
                    word_count=chunk_metadata['word_count'],
                    readability_score=chunk_metadata['readability_score']
                )
                chunks.append((chunk, metadata))
                
        return chunks
    
    def _extract_sections_enhanced(self, soup: BeautifulSoup, filing_info: Dict) -> Dict[str, Dict]:
        """Enhanced section extraction with structure analysis"""
        sections = {}
        text = soup.get_text()
        
        # Extract table of contents if available
        toc_sections = self._extract_table_of_contents(soup)
        
        # Use ToC sections if found, otherwise use pattern matching
        if toc_sections:
            sections.update(toc_sections)
        else:
            # Pattern-based extraction
            for section_name, pattern in self.section_patterns.items():
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    start_pos = matches[0].start()
                    
                    # Find next section or end of document
                    end_pos = len(text)
                    for other_name, other_pattern in self.section_patterns.items():
                        if other_name != section_name:
                            next_matches = re.search(other_pattern, text[start_pos + 100:], re.IGNORECASE)
                            if next_matches:
                                candidate_end = start_pos + 100 + next_matches.start()
                                if candidate_end < end_pos:
                                    end_pos = candidate_end
                    
                    section_text = text[start_pos:end_pos]
                    cleaned_text = self._clean_text_enhanced(section_text)
                    
                    sections[section_name] = {
                        'text': cleaned_text,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'tables': self._extract_tables_from_section(soup, start_pos, end_pos),
                        'entities': self._extract_entities(cleaned_text) if self.nlp else []
                    }
        
        # If no sections found, treat entire document as one section
        if not sections:
            cleaned_text = self._clean_text_enhanced(text)
            sections['Full Document'] = {
                'text': cleaned_text,
                'start_pos': 0,
                'end_pos': len(text),
                'tables': [],
                'entities': self._extract_entities(cleaned_text) if self.nlp else []
            }
            
        return sections
    
    def _extract_table_of_contents(self, soup: BeautifulSoup) -> Dict[str, Dict]:
        """Extract sections based on table of contents"""
        sections = {}
        
        # Look for table of contents patterns
        toc_patterns = [
            r'table\s*of\s*contents',
            r'index\s*to\s*financial\s*statements',
            r'part\s*i\s*financial\s*information'
        ]
        
        # This would involve more sophisticated HTML parsing
        # Implementation would depend on specific SEC filing formats
        # For now, return empty dict but framework is in place
        
        return sections
    
    def _extract_tables_from_section(self, soup: BeautifulSoup, start_pos: int, end_pos: int) -> List[Dict]:
        """Extract structured table data from section"""
        tables = []
        
        # Find HTML tables in the section
        html_tables = soup.find_all('table')
        
        for table in html_tables:
            try:
                # Convert table to structured data
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                
                if rows:
                    tables.append({
                        'rows': rows,
                        'header': rows[0] if rows else [],
                        'data': rows[1:] if len(rows) > 1 else []
                    })
                    
            except Exception as e:
                logger.warning(f"Error extracting table: {e}")
        
        return tables
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using spaCy"""
        if not self.nlp or len(text) > 100000:  # Skip very long texts
            return []
        
        try:
            doc = self.nlp(text[:10000])  # Limit text length for performance
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'MONEY', 'PERCENT', 'DATE', 'GPE', 'CARDINAL']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            
            return entities
            
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
            return []
    
    def _clean_text_enhanced(self, text: str) -> str:
        """Enhanced text cleaning with preservation of structure"""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        
        # Remove control characters but preserve basic formatting
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Clean up common SEC filing artifacts
        text = re.sub(r'Table of Contents', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Remove page numbers
        
        return text.strip()
    
    def _chunk_text_enhanced(self, text: str, section_data: Dict) -> List[Tuple[str, Dict]]:
        """Enhanced text chunking with metadata"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Calculate chunk metadata
            word_count = len(chunk_text.split())
            readability_score = flesch_reading_ease(chunk_text) if chunk_text.strip() else 0
            
            chunk_metadata = {
                'word_count': word_count,
                'readability_score': readability_score,
                'token_count': len(chunk_tokens),
                'has_tables': len(section_data.get('tables', [])) > 0,
                'entity_count': len(section_data.get('entities', []))
            }
            
            chunks.append((chunk_text, chunk_metadata))
            
        return chunks

class EnhancedQueryProcessor:
    """Enhanced query processing with NLP and intent analysis"""
    
    def __init__(self):
        # Expanded ticker database
        self.known_tickers = self._load_comprehensive_tickers()
        
        # Enhanced filing type mappings
        self.filing_types = {
            '10-k': '10-K', '10k': '10-K', 'annual': '10-K', 'annual report': '10-K',
            '10-q': '10-Q', '10q': '10-Q', 'quarterly': '10-Q', 'quarterly report': '10-Q',
            '8-k': '8-K', '8k': '8-K', 'current report': '8-K', 'material event': '8-K',
            'proxy': 'DEF 14A', 'def 14a': 'DEF 14A', 'proxy statement': 'DEF 14A',
            'form 3': 'FORM 3', 'form 4': 'FORM 4', 'form 5': 'FORM 5',
            'insider': ['FORM 3', 'FORM 4', 'FORM 5']
        }
        
        # Query intent patterns
        self.intent_patterns = {
            'comparison': [r'compare', r'versus', r'vs\.?', r'difference', r'contrast'],
            'trend': [r'trend', r'over time', r'evolution', r'change', r'growth'],
            'risk_analysis': [r'risk', r'factor', r'threat', r'challenge', r'concern'],
            'financial_metrics': [r'revenue', r'income', r'profit', r'margin', r'expense'],
            'governance': [r'compensation', r'executive', r'board', r'director', r'governance'],
            'strategy': [r'strategy', r'initiative', r'plan', r'approach', r'positioning'],
            'performance': [r'performance', r'results', r'achievement', r'success'],
            'regulatory': [r'regulation', r'compliance', r'legal', r'requirement'],
            'market': [r'market', r'industry', r'sector', r'competition', r'competitive']
        }
        
        # Load spaCy for enhanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
    
    def _load_comprehensive_tickers(self) -> Dict[str, str]:
        """Load comprehensive ticker database"""
        # In production, this would load from a comprehensive database
        return {
            # Technology
            'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.', 'TSLA': 'Tesla Inc.', 'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation', 'NFLX': 'Netflix Inc.', 'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.', 'ORCL': 'Oracle Corporation', 'IBM': 'IBM Corporation',
            
            # Financial Services
            'JPM': 'JPMorgan Chase & Co.', 'BAC': 'Bank of America Corp', 'WFC': 'Wells Fargo & Co',
            'GS': 'Goldman Sachs Group Inc.', 'MS': 'Morgan Stanley', 'C': 'Citigroup Inc.',
            'AXP': 'American Express Co.', 'V': 'Visa Inc.', 'MA': 'Mastercard Inc.',
            
            # Healthcare
            'JNJ': 'Johnson & Johnson', 'PFE': 'Pfizer Inc.', 'UNH': 'UnitedHealth Group Inc.',
            'ABBV': 'AbbVie Inc.', 'MRK': 'Merck & Co. Inc.', 'TMO': 'Thermo Fisher Scientific',
            
            # Consumer
            'WMT': 'Walmart Inc.', 'PG': 'Procter & Gamble Co.', 'KO': 'Coca-Cola Co.',
            'PEP': 'PepsiCo Inc.', 'NKE': 'Nike Inc.', 'MCD': 'McDonald\'s Corp.',
            
            # Energy
            'XOM': 'Exxon Mobil Corp.', 'CVX': 'Chevron Corp.', 'COP': 'ConocoPhillips'
        }
    
    def parse_query(self, query: str) -> QueryContext:
        """Enhanced query parsing with NLP analysis"""
        query_lower = query.lower()
        
        # Extract components
        tickers = self._extract_tickers_enhanced(query)
        time_periods = self._extract_time_periods_enhanced(query)
        filing_types = self._extract_filing_types_enhanced(query)
        
        # Extract entities and intent
        entities = self._extract_query_entities(query) if self.nlp else []
        intent_keywords = self._analyze_intent(query)
        
        # Calculate complexity score
        complexity_score = self._calculate_query_complexity(query, tickers, time_periods, filing_types)
        
        # Determine query type
        query_type = self._determine_query_type_enhanced(tickers, time_periods, filing_types, intent_keywords)
        
        return QueryContext(
            tickers=tickers,
            time_periods=time_periods,
            filing_types=filing_types,
            query_type=query_type,
            original_query=query,
            entities=entities,
            intent_keywords=intent_keywords,
            complexity_score=complexity_score
        )
    
    def _extract_tickers_enhanced(self, query: str) -> List[str]:
        """Enhanced ticker extraction with company name matching"""
        tickers = []
        query_upper = query.upper()
        
        # Direct ticker matching
        for ticker, company in self.known_tickers.items():
            if ticker in query_upper:
                tickers.append(ticker)
            # Match company names
            elif company.upper() in query.upper():
                tickers.append(ticker)
            # Match partial company names
            elif any(word in query.upper() for word in company.upper().split() if len(word) > 3):
                tickers.append(ticker)
        
        # Pattern matching for unknown tickers
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        found_tickers = re.findall(ticker_pattern, query.upper())
        for ticker in found_tickers:
            if len(ticker) >= 2 and ticker not in tickers:
                # Validate ticker format (not common words)
                if ticker not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE']:
                    tickers.append(ticker)
                
        return list(set(tickers))  # Remove duplicates
    
    def _extract_time_periods_enhanced(self, query: str) -> List[str]:
        """Enhanced time period extraction"""
        time_periods = []
        
        # Year patterns (including ranges)
        year_pattern = r'\b(20\d{2})(?:\s*[-–]\s*(20\d{2}))?\b'
        year_matches = re.findall(year_pattern, query)
        for match in year_matches:
            if match[1]:  # Range
                time_periods.append(f"{match[0]}-{match[1]}")
            else:  # Single year
                time_periods.append(match[0])
        
        # Quarter patterns
        quarter_patterns = [
            r'\b(Q[1-4]\s*20\d{2})\b',
            r'\b(20\d{2}\s*Q[1-4])\b',
            r'\b(first|second|third|fourth)\s*quarter\s*(20\d{2})\b',
            r'\b(Q[1-4])\s*of\s*(20\d{2})\b'
        ]
        
        for pattern in quarter_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    time_periods.append(' '.join(match))
                else:
                    time_periods.append(match)
        
        # Relative time periods
        relative_patterns = {
            'recent': r'\b(recent|latest|current|newest)\b',
            'last_year': r'\b(last|previous|prior)\s+year\b',
            'last_quarter': r'\b(last|previous|prior)\s+quarter\b',
            'ytd': r'\byear.to.date\b|ytd',
            'trailing': r'\btrailing\s+(\d+)\s+(year|quarter|month)s?\b'
        }
        
        for period_type, pattern in relative_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                time_periods.append(period_type)
                
        return time_periods
    
    def _extract_filing_types_enhanced(self, query: str) -> List[str]:
        """Enhanced filing type extraction"""
        filing_types = []
        query_lower = query.lower()
        
        for pattern, filing_type in self.filing_types.items():
            if pattern in query_lower:
                if isinstance(filing_type, list):
                    filing_types.extend(filing_type)
                else:
                    filing_types.append(filing_type)
        
        # Context-based inference
        if any(word in query_lower for word in ['insider', 'trading', 'ownership']):
            filing_types.extend(['FORM 3', 'FORM 4', 'FORM 5'])
        
        if any(word in query_lower for word in ['compensation', 'executive', 'proxy']):
            filing_types.append('DEF 14A')
            
        return list(set(filing_types))  # Remove duplicates
    
    def _extract_query_entities(self, query: str) -> List[Dict]:
        """Extract entities from query using spaCy"""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(query)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"Error extracting query entities: {e}")
            return []
    
    def _analyze_intent(self, query: str) -> List[str]:
        """Analyze query intent using pattern matching"""
        intent_keywords = []
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    intent_keywords.append(intent)
                    break  # Don't add the same intent multiple times
        
        return intent_keywords
    
    def _calculate_query_complexity(self, query: str, tickers: List[str], 
                                  time_periods: List[str], filing_types: List[str]) -> float:
        """Calculate query complexity score"""
        complexity = 0.0
        
        # Base complexity from query length
        complexity += min(len(query.split()) / 50.0, 1.0)
        
        # Multiple tickers increase complexity
        if len(tickers) > 1:
            complexity += 0.3
        
        # Multiple time periods increase complexity
        if len(time_periods) > 1:
            complexity += 0.2
        
        # Multiple filing types increase complexity
        if len(filing_types) > 1:
            complexity += 0.2
        
        # Question words indicate complexity
        question_patterns = ['what', 'how', 'why', 'when', 'where', 'which', 'compare', 'analyze']
        for pattern in question_patterns:
            if pattern in query.lower():
                complexity += 0.1
                break
        
        return min(complexity, 1.0)  # Cap at 1.0
    
    def _determine_query_type_enhanced(self, tickers: List[str], time_periods: List[str], 
                                     filing_types: List[str], intent_keywords: List[str]) -> str:
        """Enhanced query type determination"""
        # Multi-dimensional queries
        if len(tickers) >= 1 and time_periods and filing_types:
            return 'multi-dimensional'
        
        # Comparative queries
        if len(tickers) > 1 or 'comparison' in intent_keywords:
            return 'comparative'
        
        # Temporal queries
        if time_periods and 'trend' in intent_keywords:
            return 'temporal-trend'
        elif time_periods:
            return 'temporal'
        
        # Specific analysis types
        if 'risk_analysis' in intent_keywords:
            return 'risk-analysis'
        elif 'financial_metrics' in intent_keywords:
            return 'financial-analysis'
        elif 'governance' in intent_keywords:
            return 'governance-analysis'
        
        # Single ticker queries
        if len(tickers) == 1:
            return 'ticker-based'
        
        # General queries
        return 'general'

class AdvancedUncertaintyManager:
    """Sophisticated uncertainty management and confidence scoring"""
    
    def __init__(self):
        self.confidence_factors = {
            'source_diversity': 0.25,
            'temporal_relevance': 0.20,
            'content_quality': 0.20,
            'query_match': 0.15,
            'data_recency': 0.10,
            'source_authority': 0.10
        }
    
    def calculate_confidence(self, query_context: QueryContext, retrieved_chunks: List[Dict], 
                           generated_answer: str) -> Dict[str, Any]:
        """Calculate comprehensive confidence metrics"""
        if not retrieved_chunks:
            return {
                'overall_confidence': 0.0,
                'confidence_breakdown': {},
                'uncertainty_factors': ['No relevant sources found'],
                'recommendations': ['Try broader search terms', 'Check if companies are in database']
            }
        
        # Calculate individual confidence factors
        source_diversity = self._calculate_source_diversity(retrieved_chunks)
        temporal_relevance = self._calculate_temporal_relevance(query_context, retrieved_chunks)
        content_quality = self._calculate_content_quality(retrieved_chunks)
        query_match = self._calculate_query_match(query_context, retrieved_chunks)
        data_recency = self._calculate_data_recency(retrieved_chunks)
        source_authority = self._calculate_source_authority(retrieved_chunks)
        
        # Weighted overall confidence
        overall_confidence = (
            source_diversity * self.confidence_factors['source_diversity'] +
            temporal_relevance * self.confidence_factors['temporal_relevance'] +
            content_quality * self.confidence_factors['content_quality'] +
            query_match * self.confidence_factors['query_match'] +
            data_recency * self.confidence_factors['data_recency'] +
            source_authority * self.confidence_factors['source_authority']
        )
        
        # Identify uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(
            query_context, retrieved_chunks, overall_confidence
        )
        
        # Generate recommendations
        recommendations = self._generate_confidence_recommendations(
            overall_confidence, uncertainty_factors, query_context
        )
        
        return {
            'overall_confidence': round(overall_confidence, 3),
            'confidence_breakdown': {
                'source_diversity': round(source_diversity, 3),
                'temporal_relevance': round(temporal_relevance, 3),
                'content_quality': round(content_quality, 3),
                'query_match': round(query_match, 3),
                'data_recency': round(data_recency, 3),
                'source_authority': round(source_authority, 3)
            },
            'uncertainty_factors': uncertainty_factors,
            'recommendations': recommendations,
            'confidence_level': self._categorize_confidence(overall_confidence)
        }
    
    def _calculate_source_diversity(self, chunks: List[Dict]) -> float:
        """Calculate diversity of sources"""
        if not chunks:
            return 0.0
        
        # Count unique tickers, filing types, and dates
        unique_tickers = len(set(chunk['metadata']['ticker'] for chunk in chunks))
        unique_filing_types = len(set(chunk['metadata']['filing_type'] for chunk in chunks))
        unique_dates = len(set(chunk['metadata']['filing_date'] for chunk in chunks))
        unique_sections = len(set(chunk['metadata']['section'] for chunk in chunks))
        
        # Normalize based on total chunks
        total_chunks = len(chunks)
        diversity_score = (
            (unique_tickers / min(total_chunks, 5)) * 0.3 +
            (unique_filing_types / min(total_chunks, 4)) * 0.3 +
            (unique_dates / min(total_chunks, 10)) * 0.2 +
            (unique_sections / min(total_chunks, 8)) * 0.2
        )
        
        return min(diversity_score, 1.0)
    
    def _calculate_temporal_relevance(self, query_context: QueryContext, chunks: List[Dict]) -> float:
        """Calculate temporal relevance of sources"""
        if not chunks or not query_context.time_periods:
            return 0.8  # Default score when no temporal requirements
        
        current_year = datetime.now().year
        relevant_chunks = 0
        
        for chunk in chunks:
            filing_date = chunk['metadata']['filing_date']
            try:
                filing_year = int(filing_date.split('-')[0])
                
                # Check if filing matches query time periods
                for period in query_context.time_periods:
                    if period.isdigit() and int(period) == filing_year:
                        relevant_chunks += 1
                        break
                    elif period == 'recent' and (current_year - filing_year) <= 2:
                        relevant_chunks += 1
                        break
                    elif period == 'last_year' and filing_year == (current_year - 1):
                        relevant_chunks += 1
                        break
                        
            except (ValueError, IndexError):
                continue
        
        return relevant_chunks / len(chunks) if chunks else 0.0
    
    def _calculate_content_quality(self, chunks: List[Dict]) -> float:
        """Calculate quality of content"""
        if not chunks:
            return 0.0
        
        quality_scores = []
        
        for chunk in chunks:
            score = 0.0
            text = chunk['text']
            metadata = chunk['metadata']
            
            # Text length (optimal range)
            text_length = len(text.split())
            if 50 <= text_length <= 500:
                score += 0.3
            elif text_length > 500:
                score += 0.2
            
            # Readability (if available)
            if hasattr(metadata, 'readability_score') and metadata.readability_score:
                if 30 <= metadata.readability_score <= 60:  # Good readability range
                    score += 0.2
            
            # Section relevance
            relevant_sections = ['Business', 'Risk Factors', 'MD&A', 'Financial Statements']
            if metadata.get('section') in relevant_sections:
                score += 0.3
            
            # Information density (numbers, financial terms)
            financial_terms = ['revenue', 'income', 'profit', 'loss', 'cash', 'debt', 'equity']
            if any(term in text.lower() for term in financial_terms):
                score += 0.2
            
            quality_scores.append(min(score, 1.0))
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_query_match(self, query_context: QueryContext, chunks: List[Dict]) -> float:
        """Calculate how well chunks match the query"""
        if not chunks:
            return 0.0
        
        query_terms = set(query_context.original_query.lower().split())
        intent_terms = set()
        
        # Add terms based on intent
        for intent in query_context.intent_keywords or []:
            if intent == 'risk_analysis':
                intent_terms.update(['risk', 'factor', 'threat', 'challenge'])
            elif intent == 'financial_metrics':
                intent_terms.update(['revenue', 'income', 'profit', 'margin'])
            # Add more intent-based terms
        
        all_query_terms = query_terms.union(intent_terms)
        
        match_scores = []
        for chunk in chunks:
            text_terms = set(chunk['text'].lower().split())
            overlap = len(all_query_terms.intersection(text_terms))
            match_score = overlap / len(all_query_terms) if all_query_terms else 0
            match_scores.append(match_score)
        
        return np.mean(match_scores) if match_scores else 0.0
    
    def _calculate_data_recency(self, chunks: List[Dict]) -> float:
        """Calculate recency of data"""
        if not chunks:
            return 0.0
        
        current_date = datetime.now()
        recency_scores = []
        
        for chunk in chunks:
            try:
                filing_date = datetime.strptime(chunk['metadata']['filing_date'], '%Y-%m-%d')
                days_old = (current_date - filing_date).days
                
                # Score based on age (fresher is better)
                if days_old <= 90:  # 3 months
                    score = 1.0
                elif days_old <= 365:  # 1 year
                    score = 0.8
                elif days_old <= 730:  # 2 years
                    score = 0.6
                elif days_old <= 1095:  # 3 years
                    score = 0.4
                else:
                    score = 0.2
                
                recency_scores.append(score)
                
            except (ValueError, KeyError):
                recency_scores.append(0.3)  # Default for unparseable dates
        
        return np.mean(recency_scores) if recency_scores else 0.0
    
    def _calculate_source_authority(self, chunks: List[Dict]) -> float:
        """Calculate authority of sources"""
        if not chunks:
            return 0.0
        
        authority_scores = []
        
        for chunk in chunks:
            score = 0.8  # Base score for SEC filings (high authority)
            
            # Higher scores for certain filing types
            filing_type = chunk['metadata'].get('filing_type', '')
            if filing_type in ['10-K', '10-Q']:
                score = 0.9  # Annual and quarterly reports are most authoritative
            elif filing_type == 'DEF 14A':
                score = 0.85  # Proxy statements are also highly authoritative
            elif filing_type == '8-K':
                score = 0.8  # Current reports are timely and authoritative
            
            authority_scores.append(score)
        
        return np.mean(authority_scores) if authority_scores else 0.0
    
    def _identify_uncertainty_factors(self, query_context: QueryContext, chunks: List[Dict], 
                                    confidence: float) -> List[str]:
        """Identify factors contributing to uncertainty"""
        factors = []
        
        if confidence < 0.3:
            factors.append("Very low confidence due to poor source quality or relevance")
        elif confidence < 0.5:
            factors.append("Moderate uncertainty in answer quality")
            
        if len(chunks) < 3:
            factors.append("Limited number of relevant sources found")
        
        # Check for temporal mismatches
        if query_context.time_periods:
            current_year = datetime.now().year
            old_data = any(
                int(chunk['metadata']['filing_date'].split('-')[0]) < (current_year - 3)
                for chunk in chunks
                if chunk['metadata']['filing_date']
            )
            if old_data:
                factors.append("Some data sources are more than 3 years old")
        
        # Check for ticker coverage
        if query_context.tickers:
            found_tickers = set(chunk['metadata']['ticker'] for chunk in chunks)
            missing_tickers = set(query_context.tickers) - found_tickers
            if missing_tickers:
                factors.append(f"No data found for tickers: {', '.join(missing_tickers)}")
        
        # Check for filing type coverage
        if query_context.filing_types:
            found_types = set(chunk['metadata']['filing_type'] for chunk in chunks)
            missing_types = set(query_context.filing_types) - found_types
            if missing_types:
                factors.append(f"No data found for filing types: {', '.join(missing_types)}")
        
        return factors
    
    def _generate_confidence_recommendations(self, confidence: float, 
                                           uncertainty_factors: List[str], 
                                           query_context: QueryContext) -> List[str]:
        """Generate recommendations to improve confidence"""
        recommendations = []
        
        if confidence < 0.5:
            recommendations.append("Consider refining your query for more specific results")
        
        if "Limited number of relevant sources" in ' '.join(uncertainty_factors):
            recommendations.append("Try broadening search terms or including more companies")
        
        if any("old" in factor for factor in uncertainty_factors):
            recommendations.append("Consider focusing on more recent time periods")
        
        if query_context.complexity_score > 0.7:
            recommendations.append("Break down complex query into simpler sub-questions")
        
        if not query_context.tickers:
            recommendations.append("Specify company tickers for more targeted results")
        
        return recommendations
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"

class PerformanceOptimizer:
    """System performance optimization and monitoring"""
    
    def __init__(self):
        self.performance_cache = {}
        self.query_stats = defaultdict(list)
        self.optimization_settings = {
            'chunk_cache_size': 1000,
            'embedding_batch_size': 32,
            'max_concurrent_requests': 5,
            'cache_ttl_hours': 24
        }
    
    def optimize_retrieval(self, query_context: QueryContext) -> Dict[str, Any]:
        """Optimize retrieval based on query characteristics"""
        optimizations = {
            'use_cache': True,
            'parallel_processing': False,
            'chunk_limit': 50,
            'rerank_results': False
        }
        
        # Enable parallel processing for complex queries
        if query_context.complexity_score > 0.7:
            optimizations['parallel_processing'] = True
            optimizations['chunk_limit'] = 100
        
        # Use result reranking for comparative queries
        if query_context.query_type in ['comparative', 'multi-dimensional']:
            optimizations['rerank_results'] = True
        
        # Adjust chunk limit based on query type
        if query_context.query_type == 'general':
            optimizations['chunk_limit'] = 20
        elif len(query_context.tickers) > 3:
            optimizations['chunk_limit'] = 80
        
        return optimizations
    
    def track_performance(self, query_id: str, metrics: PerformanceMetrics):
        """Track query performance metrics"""
        self.query_stats[query_id].append({
            'timestamp': datetime.now(),
            'total_time': metrics.total_time,
            'retrieval_time': metrics.retrieval_time,
            'generation_time': metrics.generation_time,
            'chunks_processed': metrics.chunks_processed,
            'cache_hits': metrics.cache_hits,
            'api_calls': metrics.api_calls
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate system performance report"""
        if not self.query_stats:
            return {"message": "No performance data available"}
        
        all_metrics = []
        for query_metrics in self.query_stats.values():
            all_metrics.extend(query_metrics)
        
        if not all_metrics:
            return {"message": "No performance data available"}
        
        # Calculate aggregated metrics
        avg_total_time = np.mean([m['total_time'] for m in all_metrics])
        avg_retrieval_time = np.mean([m['retrieval_time'] for m in all_metrics])
        avg_generation_time = np.mean([m['generation_time'] for m in all_metrics])
        total_cache_hits = sum(m['cache_hits'] for m in all_metrics)
        total_api_calls = sum(m['api_calls'] for m in all_metrics)
        cache_hit_rate = total_cache_hits / (total_cache_hits + total_api_calls) if (total_cache_hits + total_api_calls) > 0 else 0
        
        return {
            'total_queries': len(all_metrics),
            'average_response_time': round(avg_total_time, 3),
            'average_retrieval_time': round(avg_retrieval_time, 3),
            'average_generation_time': round(avg_generation_time, 3),
            'cache_hit_rate': round(cache_hit_rate, 3),
            'total_api_calls': total_api_calls,
            'performance_trends': self._analyze_performance_trends(all_metrics)
        }
    
    def _analyze_performance_trends(self, metrics: List[Dict]) -> Dict[str, str]:
        """Analyze performance trends"""
        if len(metrics) < 10:
            return {"message": "Insufficient data for trend analysis"}
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x['timestamp'])
        
        # Compare first and last halves
        mid_point = len(sorted_metrics) // 2
        first_half_avg = np.mean([m['total_time'] for m in sorted_metrics[:mid_point]])
        second_half_avg = np.mean([m['total_time'] for m in sorted_metrics[mid_point:]])
        
        if second_half_avg < first_half_avg * 0.9:
            trend = "Improving - response times are getting faster"
        elif second_half_avg > first_half_avg * 1.1:
            trend = "Degrading - response times are getting slower"
        else:
            trend = "Stable - consistent performance"
        
        return {"response_time_trend": trend}

class ComprehensiveAnswerGenerator:
    """Enhanced answer generation with sophisticated LLM integration"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
        
        self.uncertainty_manager = AdvancedUncertaintyManager()
        
        # Enhanced prompt templates for different query types
        self.prompt_templates = {
            'comparative': self._get_comparative_template(),
            'risk-analysis': self._get_risk_analysis_template(),
            'financial-analysis': self._get_financial_template(),
            'temporal-trend': self._get_trend_analysis_template(),
            'general': self._get_general_template()
        }
    
    def generate_answer(self, query: str, context_chunks: List[Dict], 
                       query_context: QueryContext) -> Dict[str, Any]:
        """Generate comprehensive answer with uncertainty analysis"""
        start_time = time.time()
        
        try:
            # Select appropriate prompt template
            template = self.prompt_templates.get(
                query_context.query_type, 
                self.prompt_templates['general']
            )
            
            # Prepare enhanced context
            context_text = self._prepare_enhanced_context(context_chunks, query_context)
            
            # Create prompt
            prompt = template.format(
                query=query,
                context=context_text,
                query_type=query_context.query_type,
                tickers=', '.join(query_context.tickers) if query_context.tickers else 'N/A',
                time_periods=', '.join(query_context.time_periods) if query_context.time_periods else 'N/A'
            )
            
            # Generate answer (mock implementation - replace with actual OpenAI call)
            answer = self._generate_structured_answer(query, context_chunks, query_context)
            
            generation_time = time.time() - start_time
            
            # Calculate confidence and uncertainty
            confidence_analysis = self.uncertainty_manager.calculate_confidence(
                query_context, context_chunks, answer
            )
            
            # Extract sources with enhanced metadata
            sources = self._extract_enhanced_sources(context_chunks)
            
            # Generate executive summary for complex answers
            executive_summary = self._generate_executive_summary(answer, query_context)
            
            return {
                'answer': answer,
                'executive_summary': executive_summary,
                'sources': sources,
                'confidence_analysis': confidence_analysis,
                'context_chunks_used': len(context_chunks),
                'query_type': query_context.query_type,
                'generation_time': round(generation_time, 3),
                'answer_metadata': {
                    'word_count': len(answer.split()),
                    'readability_score': flesch_reading_ease(answer) if answer else 0,
                    'structure_score': self._calculate_answer_structure_score(answer)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': "I apologize, but I encountered an error generating the answer. Please try rephrasing your question or contact support.",
                'sources': [],
                'confidence_analysis': {'overall_confidence': 0.0},
                'error': str(e)
            }
    
    def _get_comparative_template(self) -> str:
        return """
You are a financial research analyst tasked with providing comparative analysis of SEC filings.

Query: {query}
Query Type: {query_type}
Companies: {tickers}
Time Periods: {time_periods}

Context from SEC Filings:
{context}

Instructions:
1. Provide a structured comparative analysis
2. Use clear section headers (## Company A vs Company B)
3. Include specific metrics and data points with citations [Source X]
4. Highlight key differences and similarities
5. Provide context for the comparisons
6. Note any limitations in the comparison due to data availability
7. Use tables or bullet points for easy comparison where appropriate

Structure your response as:
## Executive Summary
## Key Comparisons
## Detailed Analysis
## Limitations and Caveats

Answer:
        """
    
    def _get_risk_analysis_template(self) -> str:
        return """
You are a risk analysis specialist examining SEC filings for risk factors.

Query: {query}
Query Type: {query_type}
Companies: {tickers}
Time Periods: {time_periods}

Context from SEC Filings:
{context}

Instructions:
1. Categorize risks by type (Market, Operational, Financial, Regulatory, etc.)
2. Assess severity and likelihood based on company disclosures
3. Compare risk profiles across companies if multiple are analyzed
4. Identify emerging risks and changes over time
5. Provide specific citations for each risk factor [Source X]
6. Note any industry-specific or company-specific risk patterns

Structure your response as:
## Risk Summary
## Risk Categories
## Comparative Risk Analysis (if applicable)
## Risk Trends and Changes
## Recommendations

Answer:
        """
    
    def _get_financial_template(self) -> str:
        return """
You are a financial analyst examining SEC filings for financial metrics and performance.

Query: {query}
Query Type: {query_type}
Companies: {tickers}
Time Periods: {time_periods}

Context from SEC Filings:
{context}

Instructions:
1. Focus on quantitative financial data and metrics
2. Provide trend analysis over time where applicable
3. Include relevant ratios and performance indicators
4. Compare against industry benchmarks when possible
5. Explain the business context behind the numbers
6. Cite specific sources for all financial data [Source X]
7. Note any accounting changes or one-time items

Structure your response as:
## Financial Highlights
## Key Metrics Analysis
## Trend Analysis
## Performance Context
## Important Notes and Limitations

Answer:
        """
    
        def _get_trend_analysis_template(self) -> str:
        return """
You are a trend analysis specialist examining changes over time in SEC filings.

Query: {query}
Query Type: {query_type}
Companies: {tickers}
Time Periods: {time_periods}

Context from SEC Filings:
{context}

Instructions:
1. Identify clear trends and patterns over the specified time period
2. Highlight significant changes and inflection points
3. Provide quantitative evidence for trends where available
4. Compare trends across companies if multiple are analyzed
5. Note any external factors that may influence trends
6. Include visual trend descriptions (e.g., "steady increase", "sharp decline")
7. Cite specific sources for all trend data [Source X]

Structure your response as:
## Trend Overview
## Detailed Trend Analysis
## Comparative Trends (if applicable)
## External Influences
## Future Implications

Answer:
        """

    def _get_general_template(self) -> str:
        return """
You are a financial research analyst answering questions based on SEC filings.

Query: {query}
Query Type: {query_type}
Companies: {tickers}
Time Periods: {time_periods}

Context from SEC Filings:
{context}

Instructions:
1. Provide a clear, concise answer to the query
2. Support all claims with specific citations [Source X]
3. Include relevant context from the filings
4. Structure the answer logically
5. Highlight key points for quick understanding
6. Note any limitations in the available information

Structure your response as:
## Answer Summary
## Detailed Explanation
## Supporting Evidence
## Additional Context
## Limitations

Answer:
        """

    def _prepare_enhanced_context(self, context_chunks: List[Dict], query_context: QueryContext) -> str:
        """Prepare enhanced context for LLM with structured metadata"""
        context_parts = []
        
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk['metadata']
            context_parts.append(f"\n\n[Source {i} - {metadata.ticker} {metadata.filing_type} ({metadata.filing_date}) - {metadata.section}]\n")
            context_parts.append(chunk['text'])
            
        return ''.join(context_parts)

    def _generate_structured_answer(self, query: str, context_chunks: List[Dict], 
                             query_context: QueryContext) -> str:
    """Generate structured answer using actual OpenAI API"""
    try:
        template = self.prompt_templates.get(
            query_context.query_type, 
            self.prompt_templates['general']
        )
        
        context_text = self._prepare_enhanced_context(context_chunks, query_context)
        
        prompt = template.format(
            query=query,
            context=context_text,
            query_type=query_context.query_type,
            tickers=', '.join(query_context.tickers) if query_context.tickers else 'N/A',
            time_periods=', '.join(query_context.time_periods) if query_context.time_periods else 'N/A'
        )
        
        # Check if OpenAI API key is available
        if not openai.api_key:
            logger.warning("OpenAI API key not set, using fallback response")
            return self._generate_fallback_answer(query, context_chunks, query_context)
        
        # Actual OpenAI API call
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a financial research analyst specializing in SEC filings analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return self._generate_fallback_answer(query, context_chunks, query_context)

    def _generate_fallback_answer(self, query: str, context_chunks: List[Dict], 
                            query_context: QueryContext) -> str:
    """Generate fallback answer when OpenAI is not available"""
    if not context_chunks:
        return "I couldn't find relevant information in the SEC filings to answer your question."
    
    # Extract key information from chunks
    sources_info = []
    for i, chunk in enumerate(context_chunks[:5], 1):
        metadata = chunk['metadata']
        sources_info.append(f"[Source {i}] {metadata.get('ticker', 'Unknown')} {metadata.get('filing_type', '')} ({metadata.get('filing_date', '')}) - {metadata.get('section', '')}")
    
    if query_context.query_type == 'comparative':
        tickers = query_context.tickers
        return f"""
## Analysis Summary
Based on SEC filings for {', '.join(tickers) if tickers else 'the requested companies'}, here are the key findings:

## Key Information Found
{chr(10).join(sources_info)}

## Note
This is a basic response. For detailed analysis, please ensure OpenAI API key is configured.

## Sources Referenced
{len(context_chunks)} relevant document sections were analyzed.
"""
    else:
        return f"""
## Answer Summary
Based on the available SEC filing data, I found relevant information across {len(context_chunks)} document sections.

## Key Sources
{chr(10).join(sources_info)}

## Note
This is a basic response. For detailed analysis with specific insights and comparisons, please ensure OpenAI API key is configured.
"""
        
        elif query_context.query_type == 'risk-analysis':
            return """
## Risk Summary
The company faces three primary risk categories: market, operational, and regulatory.

## Risk Categories
1. Market Risks (Competition, Demand Fluctuation) [Source 1]
2. Operational Risks (Supply Chain, Cybersecurity) [Source 2]
3. Regulatory Risks (Privacy Laws, Antitrust) [Source 3]

## Risk Trends and Changes
Supply chain risks have increased by 30% year-over-year...
"""
        
        # Default answer for other query types
        return f"""
## Answer Summary
Based on the analysis of SEC filings, the key findings are...

## Detailed Explanation
The documents reveal that... [Source 1, 3]. Specifically, the company reported... [Source 2].

## Supporting Evidence
- Metric 1: Value (Source)
- Metric 2: Value (Source)

## Limitations
The analysis is limited to publicly available data through {datetime.now().year-1}.
"""

    def _extract_enhanced_sources(self, context_chunks: List[Dict]) -> List[Dict]:
        """Extract enhanced source metadata with relevance scoring"""
        sources = []
        
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk['metadata']
            sources.append({
                'source_id': i,
                'ticker': metadata.ticker,
                'company_name': metadata.company_name,
                'filing_type': metadata.filing_type,
                'filing_date': metadata.filing_date,
                'section': metadata.section,
                'relevance_score': self._calculate_chunk_relevance(chunk),
                'key_excerpts': self._extract_key_excerpts(chunk['text'])
            })
            
        return sorted(sources, key=lambda x: x['relevance_score'], reverse=True)

    def _calculate_chunk_relevance(self, chunk: Dict) -> float:
        """Calculate relevance score for a chunk"""
        # Simple implementation - in production would use more sophisticated scoring
        score = 0.5  # Base score
        
        # Increase for financial sections
        if chunk['metadata'].section in ['Financial Statements', 'MD&A']:
            score += 0.2
            
        # Increase for recent filings
        try:
            filing_date = datetime.strptime(chunk['metadata'].filing_date, '%Y-%m-%d')
            days_old = (datetime.now() - filing_date).days
            if days_old < 365:
                score += 0.1
        except:
            pass
            
        return min(score, 1.0)

    def _extract_key_excerpts(self, text: str) -> List[str]:
        """Extract key excerpts from chunk text"""
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) <= 3:
            return sentences
            
        # Select most informative sentences (simple implementation)
        key_sentences = []
        financial_terms = ['revenue', 'growth', 'risk', 'income', 'expense']
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in financial_terms):
                key_sentences.append(sentence)
                if len(key_sentences) >= 3:
                    break
                    
        return key_sentences if key_sentences else sentences[:3]

    def _generate_executive_summary(self, answer: str, query_context: QueryContext) -> str:
        """Generate executive summary from full answer"""
        # Simple implementation - in production would use LLM to summarize
        if "## Executive Summary" in answer:
            return answer.split("## Executive Summary")[1].split("##")[0].strip()
        
        if "## Answer Summary" in answer:
            return answer.split("## Answer Summary")[1].split("##")[0].strip()
            
        return answer[:500] + "..." if len(answer) > 500 else answer

    def _calculate_answer_structure_score(self, answer: str) -> float:
        """Calculate structure quality score for answer"""
        structure_markers = ['##', '**', '- ', '* ']
        score = 0.0
        
        for marker in structure_markers:
            if marker in answer:
                score += 0.2
                
        return min(score, 1.0)

class SECQACompleteSystem:
    """Complete SEC QA System integrating all components"""
    
    def __init__(self, config: Dict = None):
        # Initialize all components
        self.data_fetcher = EnhancedSECDataFetcher()
        self.document_processor = EnhancedDocumentProcessor()
        self.query_processor = EnhancedQueryProcessor()
        self.answer_generator = ComprehensiveAnswerGenerator()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Initialize vector database
        self.vector_db = self._initialize_vector_db()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # System configuration
        self.config = config or {
            'max_chunks': 50,
            'min_confidence': 0.4,
            'cache_enabled': True
        }
        
        # Performance tracking
        self.metrics = defaultdict(list)
        
    def _initialize_vector_db(self):
        """Initialize ChromaDB vector database"""
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=".chromadb"
        ))
        
        try:
            collection = client.get_collection("sec_filings")
            logger.info("Loaded existing SEC filings collection")
        except:
            collection = client.create_collection("sec_filings")
            logger.info("Created new SEC filings collection")
            
        return collection

    def setup_system(self, tickers: List[str] = None):
        """Setup system by downloading and processing initial data"""
        if not tickers:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Default tickers
        
        logger.info(f"Setting up system with tickers: {tickers}")
        
        all_chunks = []
        for ticker in tickers:
            logger.info(f"Processing {ticker}...")
            
            try:
                # Get recent filings
                filings = self.data_fetcher.get_company_filings(ticker, limit=3)
                
                for filing in filings:
                    content = self.data_fetcher.download_filing_content(filing)
                    if content:
                        chunks = self.document_processor.process_filing(content, filing)
                        all_chunks.extend(chunks)
                        
                        # Process in batches to avoid memory issues
                        if len(all_chunks) >= 50:
                            self._populate_vector_db(all_chunks)
                            all_chunks = []
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                continue
        
        # Process remaining chunks
        if all_chunks:
            self._populate_vector_db(all_chunks)
        
        logger.info("System setup complete")

    def _populate_vector_db(self, chunks: List[Tuple[str, DocumentMetadata]]):
        """Actually store embeddings in ChromaDB"""
        if not chunks:
            return
            
        try:
            texts = [chunk[0] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            
            # Generate unique IDs and metadata
            ids = [f"{chunk[1].ticker}_{chunk[1].chunk_id}_{hash(chunk[0])}" for chunk in chunks]
            metadatas = [asdict(chunk[1]) for chunk in chunks]
            
            # Add to vector database
            self.vector_db.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector database")
            
        except Exception as e:
            logger.error(f"Error populating vector database: {e}")

    def _retrieve_from_vector_db(self, query: str, n_results: int = 50) -> List[Dict]:
        """Retrieve similar chunks from vector database"""
        try:
            query_embedding = self.embedding_model.encode([query])
            
            results = self.vector_db.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(n_results, 100)  # ChromaDB limit
            )
            
            chunks = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    chunks.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i] if results['distances'][0] else 0.5
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving from vector database: {e}")
            return []

    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Complete query processing pipeline"""
        start_time = time.time()
        query_id = hashlib.md5(query.encode()).hexdigest()[:8]
        
        try:
            # Step 1: Parse and analyze query
            parse_start = time.time()
            query_context = self.query_processor.parse_query(query)
            parse_time = time.time() - parse_start
            
            # Step 2: Retrieve relevant documents
            retrieval_start = time.time()
            context_chunks = self._retrieve_relevant_chunks(query, query_context)
            retrieval_time = time.time() - retrieval_start
            
            # Step 3: Generate answer
            generation_start = time.time()
            answer_result = self.answer_generator.generate_answer(query, context_chunks, query_context)
            generation_time = time.time() - generation_start
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Track performance metrics
            metrics = PerformanceMetrics(
                query_start_time=start_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                chunks_processed=len(context_chunks),
                cache_hits=0,  # Would track actual cache hits in production
                api_calls=0    # Would track actual API calls in production
            )
            self.performance_optimizer.track_performance(query_id, metrics)
            
            # Prepare final result
            result = {
                'query': query,
                'query_id': query_id,
                'query_context': asdict(query_context),
                'answer': answer_result['answer'],
                'executive_summary': answer_result.get('executive_summary', ''),
                'sources': answer_result['sources'],
                'confidence_analysis': answer_result['confidence_analysis'],
                'performance_metrics': {
                    'parse_time': round(parse_time, 3),
                    'retrieval_time': round(retrieval_time, 3),
                    'generation_time': round(generation_time, 3),
                    'total_time': round(total_time, 3)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'error': str(e),
                'query_id': query_id,
                'answer': "An error occurred while processing your query. Please try again."
            }
    
    def _retrieve_relevant_chunks(self, query: str, query_context: QueryContext) -> List[Dict]:
    """Retrieve relevant document chunks for a query"""
    try:
        # First try to retrieve from vector database
        chunks_from_db = self._retrieve_from_vector_db(query, n_results=100)
        
        if chunks_from_db:
            # Filter by query context
            filtered_chunks = self._filter_chunks_by_context_dict(chunks_from_db, query_context)
            if filtered_chunks:
                return filtered_chunks[:self.config['max_chunks']]
        
        # Fallback: fetch and process new documents
        logger.info("No relevant chunks in database, fetching new documents...")
        
        filings = []
        for ticker in query_context.tickers or ['AAPL']:  # Default to AAPL if no tickers
            ticker_filings = self.data_fetcher.get_company_filings(
                ticker,
                filing_types=query_context.filing_types,
                start_date=self._get_start_date(query_context.time_periods),
                limit=5
            )
            filings.extend(ticker_filings)
        
        # Process filings to get chunks
        all_chunks = []
        for filing in filings[:5]:  # Limit to 5 filings
            content = self.data_fetcher.download_filing_content(filing)
            if content:
                chunks = self.document_processor.process_filing(content, filing)
                all_chunks.extend(chunks)
        
        if all_chunks:
            # Add to vector database for future use
            self._populate_vector_db(all_chunks)
            
            # Filter and score chunks
            filtered_chunks = self._filter_chunks_by_context(all_chunks, query_context)
            
            # Convert to dict format and calculate similarity
            query_embedding = self.embedding_model.encode(query)
            scored_chunks = []
            
            for chunk_text, chunk_metadata in filtered_chunks:
                chunk_embedding = self.embedding_model.encode(chunk_text)
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                
                scored_chunks.append({
                    'text': chunk_text,
                    'metadata': asdict(chunk_metadata),
                    'similarity_score': similarity
                })
            
            scored_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            return scored_chunks[:self.config['max_chunks']]
        
        return []
        
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {e}")
        return []

def _filter_chunks_by_context_dict(self, chunks: List[Dict], query_context: QueryContext) -> List[Dict]:
    """Filter chunks when they come from vector DB as dicts"""
    filtered = []
    
    for chunk in chunks:
        metadata = chunk['metadata']
        
        # Filter by ticker
        if query_context.tickers and metadata.get('ticker') not in query_context.tickers:
            continue
            
        # Filter by filing type
        if query_context.filing_types and metadata.get('filing_type') not in query_context.filing_types:
            continue
            
        # Filter by time period
        if query_context.time_periods and not self._matches_time_period(
            metadata.get('filing_date', ''), query_context.time_periods
        ):
            continue
            
        filtered.append(chunk)
    
    return filtered
    
    def _matches_time_period(self, filing_date: str, time_periods: List[str]) -> bool:
        """Check if filing date matches any time period"""
        if not time_periods:
            return True
            
        try:
            filing_year = int(filing_date.split('-')[0])
        except:
            return False
            
        for period in time_periods:
            if period.isdigit() and len(period) == 4:
                if filing_year == int(period):
                    return True
            elif '-' in period:
                try:
                    start, end = period.split('-')
                    if len(start) == 4 and len(end) == 4:
                        if int(start) <= filing_year <= int(end):
                            return True
                except:
                    continue
            elif period == 'recent':
                if (datetime.now().year - filing_year) <= 2:
                    return True
            elif period == 'last_year':
                if filing_year == (datetime.now().year - 1):
                    return True
                    
        return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get system performance report"""
        return self.performance_optimizer.get_performance_report()
    
    def evaluate_sample_questions(self) -> Dict[str, Any]:
        """Evaluate system against sample questions"""
        sample_questions = [
            "What are the primary revenue drivers for major technology companies, and how have they evolved?",
            "Compare R&D spending trends across companies. What insights about innovation investment strategies?",
            "Identify significant working capital changes for financial services companies and driving factors.",
            "What are the most commonly cited risk factors across industries? How do same-sector companies prioritize differently?",
            "How do companies describe climate-related risks? Notable industry differences?",
            "Analyze recent executive compensation changes. What trends emerge?",
            "What significant insider trading activity occurred? What might this indicate?",
            "How are companies positioning regarding AI and automation? Strategic approaches?",
            "Identify recent M&A activity. What strategic rationale do companies provide?",
            "How do companies describe competitive advantages? What themes emerge?"
        ]
        
        results = []
        for question in sample_questions:
            result = self.process_query(question)
            results.append({
                'question': question,
                'answer_summary': result.get('executive_summary', ''),
                'confidence': result.get('confidence_analysis', {}).get('overall_confidence', 0),
                'processing_time': result.get('performance_metrics', {}).get('total_time', 0)
            })
            
        return {
            'total_questions': len(results),
            'average_confidence': round(np.mean([r['confidence'] for r in results]), 3),
            'average_time': round(np.mean([r['processing_time'] for r in results]), 3),
            'detailed_results': results
        }

# Example usage
if __name__ == "__main__":
    import os
    
    # Set OpenAI API key (get from environment or set directly)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        openai.api_key = openai_api_key
        print("✓ OpenAI API key loaded")
    else:
        print("⚠ OpenAI API key not found. Set OPENAI_API_KEY environment variable or system will use fallback responses.")
    
    # Initialize system
    print("Initializing SEC QA System...")
    sec_qa_system = SECQACompleteSystem()
    
    # Setup system with sample companies (this will take a few minutes)
    print("Setting up system with sample companies...")
    sec_qa_system.setup_system(['AAPL', 'MSFT'])
    
    # Process a sample query
    sample_query = "Compare Apple and Microsoft R&D spending trends"
    print(f"\nProcessing query: {sample_query}")
    result = sec_qa_system.process_query(sample_query)
    
    print("\n" + "="*80)
    print("QUERY RESULT")
    print("="*80)
    print("\nQuery:", result.get('query', 'N/A'))
    print("\nAnswer:", result.get('answer', 'N/A'))
    print("\nConfidence:", result.get('confidence_analysis', {}).get('overall_confidence', 'N/A'))
    
    sources = result.get('sources', [])
    if sources:
        print(f"\nTop {min(3, len(sources))} Sources:")
        for i, source in enumerate(sources[:3], 1):
            print(f"{i}. {source.get('ticker', 'N/A')} {source.get('filing_type', 'N/A')} ({source.get('filing_date', 'N/A')})")
    
    print(f"\nProcessing Time: {result.get('total_time', 'N/A')} seconds")
    print("="*80)
