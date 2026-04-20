"""
Web Search Integration for RAG
Provides real-time web search results to augment RAG responses
Supports: Tavily, Serper, DuckDuckGo (free)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str
    score: float


class WebSearchProvider:
    """Unified interface for web search providers"""
    
    def __init__(self, settings):
        self.settings = settings
    
    def search(self, query: str, num_results: int = 5) -> list[WebSearchResult]:
        """Search the web using configured provider"""
        
        if not self.settings.enable_web_search:
            return []
        
        # Try Tavily first (best for AI/RAG)
        if self.settings.tavily_api_key:
            try:
                return self._search_tavily(query, num_results)
            except Exception as e:
                print(f"Tavily search failed: {e}")
        
        # Fallback to DuckDuckGo (free, no API key)
        try:
            return self._search_duckduckgo(query, num_results)
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
        
        return []
    
    def _search_tavily(self, query: str, num_results: int) -> list[WebSearchResult]:
        """Search using Tavily API (best for AI)"""
        
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "max_results": num_results,
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
        }
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        
        # Add AI answer if available
        if data.get("answer"):
            results.append(WebSearchResult(
                title="AI Summary",
                url="",
                content=data["answer"],
                score=1.0,
            ))
        
        # Add web results
        for item in data.get("results", []):
            results.append(WebSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.5),
            ))
        
        return results
    
    def _search_duckduckgo(self, query: str, num_results: int) -> list[WebSearchResult]:
        """Search using DuckDuckGo (free, no API key)"""
        
        # Using HTML scraper approach
        url = "https://html.duckduckgo.com/html/"
        
        payload = {
            "q": query,
            "b": "",
        }
        
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        results = []
        
        for result in soup.select(".result"):
            title_elem = result.select_one(".result__title")
            link_elem = result.select_one(".result__url")
            snippet_elem = result.select_one(".result__snippet")
            
            if title_elem and link_elem:
                # Get the actual URL
                href = title_elem.find("a")
                if href:
                    url = href.get("href", "")
                    # DuckDuckGo redirects through their own URL
                    if "uddg=" in url:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                        url = parsed.get("uddg", [""])[0]
                    
                    results.append(WebSearchResult(
                        title=title_elem.get_text(strip=True),
                        url=url,
                        content=snippet_elem.get_text(strip=True) if snippet_elem else "",
                        score=0.8,
                    ))
            
            if len(results) >= num_results:
                break
        
        return results
    
    def format_for_rag(self, results: list[WebSearchResult]) -> str:
        """Format search results as context for RAG"""
        
        if not results:
            return ""
        
        blocks = []
        for i, r in enumerate(results, 1):
            if r.url:  # Skip AI summary without URL
                blocks.append(f"[{i}] {r.title}\n{r.url}\n{r.content}")
            else:
                blocks.append(f"[AI Summary]\n{r.content}")
        
        return "\n\n".join(blocks)


def get_web_search(settings) -> WebSearchProvider:
    return WebSearchProvider(settings)
