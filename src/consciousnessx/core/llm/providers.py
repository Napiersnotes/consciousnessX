"""
LLM Provider Factory and Implementations
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import openai
import anthropic
import google.generativeai as genai
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from ..config import settings, LLMProvider


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Any = None


class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the client for the specific provider"""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass
    
    def calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate the cost of the API call"""
        return 0.0  # Default implementation


class OpenAIProvider(BaseLLMProvider):
    def _initialize_client(self):
        api_key = self.config.get("api_key") or settings.OPENAI_API_KEY
        base_url = self.config.get("base_url")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> AsyncGenerator[LLMResponse, None]:
        model = self.config.get("model", settings.OPENAI_MODEL)
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            
            if stream:
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield LLMResponse(
                            content=chunk.choices[0].delta.content,
                            model=model,
                            usage={},
                            finish_reason="",
                            raw_response=chunk,
                        )
            else:
                yield LLMResponse(
                    content=response.choices[0].message.content,
                    model=model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    finish_reason=response.choices[0].finish_reason,
                    raw_response=response,
                )
                
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    def calculate_cost(self, usage: Dict[str, int]) -> float:
        # GPT-4 Turbo pricing (approx)
        prompt_cost_per_token = 0.01 / 1000  # $0.01 per 1K tokens
        completion_cost_per_token = 0.03 / 1000  # $0.03 per 1K tokens
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        return (
            prompt_tokens * prompt_cost_per_token +
            completion_tokens * completion_cost_per_token
        )


class AnthropicProvider(BaseLLMProvider):
    def _initialize_client(self):
        api_key = self.config.get("api_key") or settings.ANTHROPIC_API_KEY
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> AsyncGenerator[LLMResponse, None]:
        model = self.config.get("model", settings.ANTHROPIC_MODEL)
        max_tokens = max_tokens or 4096
        
        # Convert to Anthropic format
        system_prompt = None
        converted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                converted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        try:
            response = await self.client.messages.create(
                model=model,
                messages=converted_messages,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            
            if stream:
                async for chunk in response:
                    if chunk.type == "content_block_delta":
                        yield LLMResponse(
                            content=chunk.delta.text,
                            model=model,
                            usage={},
                            finish_reason="",
                            raw_response=chunk,
                        )
            else:
                content = "".join(
                    block.text for block in response.content
                    if hasattr(block, 'text')
                )
                
                yield LLMResponse(
                    content=content,
                    model=model,
                    usage={
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    finish_reason=response.stop_reason,
                    raw_response=response,
                )
                
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        # Anthropic doesn't have a public embedding API yet
        # Fallback to local embeddings or OpenAI
        raise NotImplementedError("Anthropic embeddings not implemented")
    
    def calculate_cost(self, usage: Dict[str, int]) -> float:
        # Claude 3 Opus pricing (approx)
        input_cost_per_token = 0.015 / 1000  # $15 per 1M tokens
        output_cost_per_token = 0.075 / 1000  # $75 per 1M tokens
        
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        return (
            input_tokens * input_cost_per_token +
            output_tokens * output_cost_per_token
        )


class GoogleProvider(BaseLLMProvider):
    def _initialize_client(self):
        api_key = self.config.get("api_key") or settings.GOOGLE_API_KEY
        genai.configure(api_key=api_key)
        self.model_name = self.config.get("model", settings.GOOGLE_MODEL)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> AsyncGenerator[LLMResponse, None]:
        model = genai.GenerativeModel(self.model_name)
        
        # Convert to Google format
        formatted_content = []
        for msg in messages:
            formatted_content.append(f"{msg['role']}: {msg['content']}")
        
        prompt = "\n".join(formatted_content)
        
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens or 2048,
            }
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=generation_config,
                stream=stream,
            )
            
            if stream:
                async for chunk in response:
                    if chunk.text:
                        yield LLMResponse(
                            content=chunk.text,
                            model=self.model_name,
                            usage={},
                            finish_reason="",
                            raw_response=chunk,
                        )
            else:
                yield LLMResponse(
                    content=response.text,
                    model=self.model_name,
                    usage={},
                    finish_reason="",
                    raw_response=response,
                )
                
        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            raise
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            # Using textembedding-gecko model
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(result["embedding"])
            return embeddings
        except Exception as e:
            logger.error(f"Google embedding error: {str(e)}")
            raise


class LocalProvider(BaseLLMProvider):
    def _initialize_client(self):
        import httpx
        self.base_url = self.config.get("base_url", settings.OLLAMA_BASE_URL)
        self.model = self.config.get("model", settings.OLLAMA_MODEL)
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> AsyncGenerator[LLMResponse, None]:
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or 2048,
            },
            "stream": stream,
        }
        
        try:
            if stream:
                async with self.client.stream("POST", url, json=payload) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    yield LLMResponse(
                                        content=data["message"]["content"],
                                        model=self.model,
                                        usage={},
                                        finish_reason="",
                                        raw_response=data,
                                    )
                            except json.JSONDecodeError:
                                continue
            else:
                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                yield LLMResponse(
                    content=data["message"]["content"],
                    model=self.model,
                    usage={},
                    finish_reason=data.get("done_reason", ""),
                    raw_response=data,
                )
                
        except Exception as e:
            logger.error(f"Local model error: {str(e)}")
            raise
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/api/embeddings"
        
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model,
                "prompt": text,
            }
            
            try:
                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])
            except Exception as e:
                logger.error(f"Local embedding error: {str(e)}")
                embeddings.append([])
        
        return embeddings


class LLMFactory:
    """Factory for creating LLM providers"""
    
    _providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.GOOGLE: GoogleProvider,
        LLMProvider.LOCAL: LocalProvider,
    }
    
    @classmethod
    def create_provider(
        cls,
        provider: LLMProvider = None,
        config: Dict[str, Any] = None,
    ) -> BaseLLMProvider:
        """Create an LLM provider instance"""
        provider = provider or settings.DEFAULT_LLM_PROVIDER
        
        if provider not in cls._providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        provider_config = config or settings.LLM_CONFIG.get(provider, {})
        return cls._providers[provider](provider_config)
    
    @classmethod
    def get_available_providers(cls) -> List[LLMProvider]:
        """Get list of available providers"""
        return list(cls._providers.keys())
