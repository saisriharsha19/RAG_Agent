import aiohttp
import asyncio
import os
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import requests
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSearcher:
    def __init__(self, fallback_enabled: bool = True):
        self.fallback_enabled = fallback_enabled
        self.search_engines = [
            self._search_duckduckgo,
            self._search_bing_fallback,
            self._search_serper_fallback
        ]
        self.current_engine_index = 0
    
    async def search(self, query: str, num_results: int = 3, user_preference: str = "auto") -> Dict[str, Any]:
        """
        Enhanced search with fallback mechanism and user preferences
        """
        search_result = {
            "results": [],
            "search_engine_used": "none",
            "fallback_attempts": 0,
            "search_quality": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        if user_preference == "disabled":
            logger.info("Web search disabled by user preference")
            return search_result
        
        # Try primary search engines with fallback
        for attempt, search_engine in enumerate(self.search_engines):
            try:
                logger.info(f"Attempting search with engine {attempt + 1}")
                results = await search_engine(query, num_results)
                
                if results:
                    search_result.update({
                        "results": results,
                        "search_engine_used": search_engine.__name__,
                        "fallback_attempts": attempt,
                        "search_quality": self._assess_search_quality(results)
                    })
                    logger.info(f"Search successful with {len(results)} results")
                    break
                    
            except Exception as e:
                logger.warning(f"Search engine {search_engine.__name__} failed: {str(e)}")
                search_result["fallback_attempts"] = attempt + 1
                
                if not self.fallback_enabled or attempt == len(self.search_engines) - 1:
                    logger.error("All search engines failed")
                    break
                    
                await asyncio.sleep(1)  # Brief delay before fallback
        
        return search_result
    
    def _assess_search_quality(self, results: List[Dict]) -> str:
        """Assess the quality of search results"""
        if not results:
            return "no_results"
        
        avg_content_length = sum(len(r.get('content', '')) for r in results) / len(results)
        
        if avg_content_length > 200:
            return "high"
        elif avg_content_length > 100:
            return "medium"
        else:
            return "low"
    
    async def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Enhanced DuckDuckGo search with multiple endpoints"""
        
        # Try instant answer API first
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Get abstract if available
                        if data.get('Abstract'):
                            results.append({
                                'title': data.get('AbstractText', 'DuckDuckGo Result')[:100],
                                'content': data.get('Abstract'),
                                'url': data.get('AbstractURL', ''),
                                'source': 'DuckDuckGo Instant',
                                'confidence': 0.9
                            })
                        
                        # Get related topics
                        for topic in data.get('RelatedTopics', [])[:num_results-1]:
                            if isinstance(topic, dict) and topic.get('Text'):
                                results.append({
                                    'title': topic.get('Text', '')[:100] + '...',
                                    'content': topic.get('Text', ''),
                                    'url': topic.get('FirstURL', ''),
                                    'source': 'DuckDuckGo Topics',
                                    'confidence': 0.7
                                })
        
        except Exception as e:
            logger.warning(f"DuckDuckGo API failed: {str(e)}")
        
        # If no good results, try HTML scraping
        if len(results) < num_results:
            scraped_results = await self._scrape_duckduckgo_html(query, num_results - len(results))
            results.extend(scraped_results)
        
        return results[:num_results]
    
    async def _scrape_duckduckgo_html(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Scrape DuckDuckGo HTML results as fallback"""
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                async with session.get(search_url, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        results = []
                        result_divs = soup.find_all('div', class_='result')[:num_results]
                        
                        for div in result_divs:
                            title_elem = div.find('a', class_='result__a')
                            snippet_elem = div.find('a', class_='result__snippet')
                            
                            if title_elem and snippet_elem:
                                results.append({
                                    'title': title_elem.get_text(strip=True),
                                    'content': snippet_elem.get_text(strip=True),
                                    'url': title_elem.get('href', ''),
                                    'source': 'DuckDuckGo HTML',
                                    'confidence': 0.6
                                })
                        
                        return results
        except Exception as e:
            logger.error(f"DuckDuckGo HTML scraping failed: {str(e)}")
        
        return []
    
    async def _search_bing_fallback(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Bing search fallback (requires API key if available)"""
        
        bing_api_key = os.getenv('BING_API_KEY')
        if not bing_api_key:
            logger.info("Bing fallback attempted - API key required for full functionality")
            
            return [{
                'title': f'Bing Search: {query}',
                'content': f'Bing search results would appear here for query: {query}. Configure BING_API_KEY for full functionality.',
                'url': f'https://www.bing.com/search?q={query}',
                'source': 'Bing Fallback',
                'confidence': 0.3
            }]
        
        # If we have a Bing API key, use it
        url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
        headers = {
            'Ocp-Apim-Subscription-Key': bing_api_key
        }
        params = {
            'q': query,
            'count': num_results,
            'textDecorations': False,
            'textFormat': 'Raw'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for item in data.get('webPages', {}).get('value', [])[:num_results]:
                            results.append({
                                'title': item.get('name', ''),
                                'content': item.get('snippet', ''),
                                'url': item.get('url', ''),
                                'source': 'Bing API',
                                'confidence': 0.8
                            })
                        
                        return results
        
        except Exception as e:
            logger.error(f"Bing API search failed: {str(e)}")
        
        return []
    
    async def _search_serper_fallback(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Serper.dev API fallback"""
        
        serper_api_key = os.getenv('SERPER_API_KEY')
        if not serper_api_key:
            logger.info("Serper fallback skipped - no API key")
            return []
        
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num_results
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for item in data.get('organic', [])[:num_results]:
                            results.append({
                                'title': item.get('title', ''),
                                'content': item.get('snippet', ''),
                                'url': item.get('link', ''),
                                'source': 'Serper/Google',
                                'confidence': 0.8
                            })
                        
                        return results
        
        except Exception as e:
            logger.error(f"Serper search failed: {str(e)}")
        
        return []
    
    async def verify_search_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Verify and enhance search results"""
        verified_results = []
        
        for result in results:
            # Basic verification
            if result.get('content') and len(result['content']) > 20:
                # Calculate relevance score
                query_words = set(query.lower().split())
                content_words = set(result['content'].lower().split())
                relevance_score = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
                
                result['relevance_score'] = relevance_score
                result['verified'] = True
                
                verified_results.append(result)
        
        # Sort by relevance and confidence
        verified_results.sort(key=lambda x: (x.get('relevance_score', 0) + x.get('confidence', 0)) / 2, reverse=True)
        
        return verified_results