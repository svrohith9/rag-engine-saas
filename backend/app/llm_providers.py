"""
Multi-Provider LLM Integration
Supports: Gemini, OpenAI, Anthropic Claude, Ollama
With streaming support via Server-Sent Events (SSE)
"""

from __future__ import annotations

import json
from typing import Optional, AsyncGenerator, Union
from dataclasses import dataclass

import requests
from anthropic import Anthropic


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None


@dataclass
class StreamChunk:
    delta: str
    model: str
    done: bool = False


class LLMProvider:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self, settings):
        self.settings = settings
    
    # ==================== Content Generation ====================
    
    def generate(
        self,
        system_text: str,
        user_text: str,
        temperature: float = 0.2,
        image_paths: list[str] = None,
    ) -> LLMResponse:
        """Generate content using the configured provider"""
        
        provider = self.settings.llm_provider
        
        if provider == "gemini":
            return self._generate_gemini(system_text, user_text, temperature, image_paths)
        elif provider == "openai":
            return self._generate_openai(system_text, user_text, temperature)
        elif provider == "anthropic":
            return self._generate_anthropic(system_text, user_text, temperature)
        elif provider == "ollama":
            return self._generate_ollama(system_text, user_text, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate_stream(
        self,
        system_text: str,
        user_text: str,
        temperature: float = 0.2,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate content with streaming"""
        
        provider = self.settings.llm_provider
        
        if provider == "openai":
            yield from self._stream_openai(system_text, user_text, temperature)
        elif provider == "anthropic":
            yield from self._stream_anthropic(system_text, user_text, temperature)
        elif provider == "ollama":
            yield from self._stream_ollama(system_text, user_text, temperature)
        elif provider == "gemini":
            # Gemini doesn't support streaming in the same way, fallback to regular
            response = self._generate_gemini(system_text, user_text, temperature, None)
            yield StreamChunk(delta=response.content, model=response.model, done=True)
        else:
            raise ValueError(f"Streaming not supported for provider: {provider}")
    
    # ==================== Gemini ====================
    
    def _generate_gemini(
        self,
        system_text: str,
        user_text: str,
        temperature: float,
        image_paths: list[str],
    ) -> LLMResponse:
        """Generate using Google Gemini"""
        import google.generativeai as genai
        
        genai.configure(api_key=self.settings.gemini_api_key)
        
        # Build content parts
        contents = []
        
        # Add user text
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        
        # Add images if provided
        if image_paths:
            for img_path in image_paths:
                try:
                    img = genai.upload_file(img_path)
                    contents[-1]["parts"].append({"file_data": {"mime_type": "image/jpeg", "file_uri": img.uri}})
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
        
        model = genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            system_instruction=system_text,
        )
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": 8192,
        }
        
        response = model.generate_content(
            contents,
            generation_config=generation_config,
        )
        
        return LLMResponse(
            content=response.text,
            model=self.settings.gemini_model,
        )
    
    # ==================== OpenAI ====================
    
    def _generate_openai(
        self,
        system_text: str,
        user_text: str,
        temperature: float,
    ) -> LLMResponse:
        """Generate using OpenAI GPT models"""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.settings.openai_api_key)
        
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=8192,
        )
        
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
        )
    
    def _stream_openai(
        self,
        system_text: str,
        user_text: str,
        temperature: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream OpenAI responses"""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.settings.openai_api_key)
        
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        
        stream = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=8192,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(
                    delta=chunk.choices[0].delta.content,
                    model=chunk.model,
                    done=False,
                )
        
        yield StreamChunk(delta="", model=self.settings.openai_model, done=True)
    
    # ==================== Anthropic ====================
    
    def _generate_anthropic(
        self,
        system_text: str,
        user_text: str,
        temperature: float,
    ) -> LLMResponse:
        """Generate using Anthropic Claude"""
        client = Anthropic(api_key=self.settings.anthropic_api_key)
        
        message = client.messages.create(
            model=self.settings.anthropic_model,
            system=system_text,
            messages=[{"role": "user", "content": user_text}],
            temperature=temperature,
            max_tokens=8192,
        )
        
        return LLMResponse(
            content=message.content[0].text,
            model=message.model,
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        )
    
    def _stream_anthropic(
        self,
        system_text: str,
        user_text: str,
        temperature: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream Anthropic responses"""
        client = Anthropic(api_key=self.settings.anthropic_api_key)
        
        with client.messages.stream(
            model=self.settings.anthropic_model,
            system=system_text,
            messages=[{"role": "user", "content": user_text}],
            temperature=temperature,
            max_tokens=8192,
        ) as stream:
            for chunk in stream:
                if chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        yield StreamChunk(
                            delta=chunk.delta.text,
                            model=self.settings.anthropic_model,
                            done=False,
                        )
                elif chunk.type == "message_stop":
                    yield StreamChunk(
                        delta="",
                        model=self.settings.anthropic_model,
                        done=True,
                    )
    
    # ==================== Ollama ====================
    
    def _generate_ollama(
        self,
        system_text: str,
        user_text: str,
        temperature: float,
    ) -> LLMResponse:
        """Generate using Ollama (local models)"""
        
        url = f"{self.settings.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.settings.ollama_model,
            "prompt": f"System: {system_text}\n\nUser: {user_text}",
            "temperature": temperature,
            "stream": False,
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        
        return LLMResponse(
            content=data.get("response", ""),
            model=self.settings.ollama_model,
        )
    
    def _stream_ollama(
        self,
        system_text: str,
        user_text: str,
        temperature: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream Ollama responses"""
        
        url = f"{self.settings.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.settings.ollama_model,
            "prompt": f"System: {system_text}\n\nUser: {user_text}",
            "temperature": temperature,
            "stream": True,
        }
        
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield StreamChunk(
                            delta=data["response"],
                            model=self.settings.ollama_model,
                            done=data.get("done", False),
                        )
    
    # ==================== Embeddings ====================
    
    def embed_text(self, text: str) -> Optional[list[float]]:
        """Get embeddings using the configured provider"""
        
        provider = self.settings.llm_provider
        
        # Try Gemini embeddings first
        if self.settings.gemini_api_key and self.settings.gemini_embed_model:
            try:
                return self._embed_gemini(text)
            except Exception as e:
                print(f"Gemini embedding failed: {e}")
        
        # Fallback to OpenAI embeddings
        if self.settings.openai_api_key:
            try:
                return self._embed_openai(text)
            except Exception as e:
                print(f"OpenAI embedding failed: {e}")
        
        return None
    
    def _embed_gemini(self, text: str) -> list[float]:
        """Get embeddings from Gemini"""
        import google.generativeai as genai
        
        genai.configure(api_key=self.settings.gemini_api_key)
        
        result = genai.embed_content(
            model=self.settings.gemini_embed_model,
            content=text,
            task_type="RETRIEVAL_QUERY",
        )
        
        return result["embedding"]
    
    def _embed_openai(self, text: str) -> list[float]:
        """Get embeddings from OpenAI"""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.settings.openai_api_key)
        
        response = client.embeddings.create(
            model=self.settings.openai_embed_model,
            input=text,
        )
        
        return response.data[0].embedding


# Provider factory
def get_provider(settings) -> LLMProvider:
    return LLMProvider(settings)
