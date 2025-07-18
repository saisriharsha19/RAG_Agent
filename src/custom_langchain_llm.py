from typing import Any, List, Optional, Dict, Union
import asyncio
from functools import partial
import logging

# LangChain core imports
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)

# NeMo Guardrails registration import
from nemoguardrails.llm.providers import register_llm_provider

# Your existing UFL client factory
from .ufl_llm_client import create_llm_client

# Set up logging
logger = logging.getLogger(__name__)


class MyCustomLLM(LLM):
    """A LangChain-compatible LLM provider for the UFL AI service."""
    
    client: Any = None
    model_name: str = "ufl-ai-model"  # Default model name
    temperature: float = 0.7
    max_tokens: int = 1000
    
    class Config:
        """Configuration for this pydantic object."""
        extra = "forbid"  # Forbid extra attributes

    def __init__(self, **kwargs: Any):
        # Extract any custom parameters before calling super
        model_name = kwargs.pop("model_name", "ufl-ai-model")
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", 1000)
        
        super().__init__(**kwargs)
        
        # Set instance attributes
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Use your factory to create the client instance
        self.client = create_llm_client()

    @property
    def _llm_type(self) -> str:
        """A unique name for your custom LLM class."""
        return "ufl_ai_langchain"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """The synchronous method for generating a response."""
        logger.info(f"_call invoked with prompt: {prompt[:50]}...")
        
        # Handle the case where we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop is running, we can use asyncio.run
            return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))
        else:
            # We're in an event loop, use nest_asyncio or thread pool
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))
            except ImportError:
                # Fallback to thread pool if nest_asyncio not available
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run, 
                        self._acall(prompt, stop, run_manager, **kwargs)
                    ).result()

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """The asynchronous method for generating a response."""
        logger.info(f"_acall invoked with prompt: {prompt[:50]}...")
        
        try:
            # Check if client has the expected method
            if not hasattr(self.client, 'generate_response'):
                raise AttributeError(
                    f"Client {type(self.client).__name__} doesn't have 'generate_response' method"
                )
            
            # Merge kwargs with instance parameters
            generation_kwargs = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            # Use your existing async client to get the response
            response_text = await self.client.generate_response(
                prompt, 
                **generation_kwargs
            )
            
            logger.info(f"Received response: {response_text[:50]}...")
            
            # Handle stop sequences if provided
            if stop:
                for stop_seq in stop:
                    if stop_seq in response_text:
                        response_text = response_text.split(stop_seq)[0]
                        break
            
            # Notify callbacks if present
            if run_manager:
                await run_manager.on_llm_new_token(response_text)
            
            return response_text
            
        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Error in _acall: {str(e)}", exc_info=True)
            raise

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model": getattr(self.client, 'model', self.model_name),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "llm_type": self._llm_type,
        }

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.
        
        This is a simple implementation. You should replace it with
        your actual tokenizer if available.
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    def get_num_tokens_from_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Get the number of tokens in a list of messages."""
        total_text = ""
        for message in messages:
            if isinstance(message, dict):
                total_text += message.get("content", "")
            else:
                total_text += str(message)
        return self.get_num_tokens(total_text)


# Factory function for creating instances
def create_custom_llm(**kwargs: Any) -> MyCustomLLM:
    """Factory function to create a MyCustomLLM instance.
    
    This can be useful for NeMo Guardrails registration.
    """
    return MyCustomLLM(**kwargs)


# Register your custom LangChain provider with NeMo Guardrails
# It will now be available under the name "custom_llm".
register_llm_provider("custom_llm", MyCustomLLM)
