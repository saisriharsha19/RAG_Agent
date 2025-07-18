import asyncio
import nest_asyncio
from typing import Any, List, Optional, Dict

# LangChain & NeMo Guardrails Imports
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from nemoguardrails.llm.providers import register_llm_provider

# --- This now imports your REAL client factory ---
from .ufl_llm_client import create_llm_client

class MyCustomLLM(LLM):
    """A production-ready LangChain-compatible LLM provider for the UFL AI service."""
    
    client: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Use your real factory to create the client instance
        self.client = create_llm_client()

    @property
    def _llm_type(self) -> str:
        return "ufl_ai_langchain"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self.client.generate_response(prompt)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.client.model}

# Register your custom provider so the config.yml can find it
register_llm_provider("custom_llm", MyCustomLLM)
print("✅ SUCCESS: The 'custom_llm' provider has been successfully registered.")