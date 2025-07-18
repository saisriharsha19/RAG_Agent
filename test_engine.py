import asyncio
import logging
from typing import Any, List, Optional, Dict

# --- LangChain & NeMo Guardrails Imports ---
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.llm.providers import register_llm_provider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- PART 1: The Custom LLM Provider (Self-Contained) ---

class DummyClient:
    """A simple, fake client for this self-contained test."""
    model = "dummy-model"
    async def generate_response(self, prompt: str) -> str:
        # Simulate your UFL client. This will return the classification.
        if "what is the email address of john doe?" in prompt.lower():
            return "request_pii"
        else:
            return "safe"

class MyCustomLLM(LLM):
    """A self-contained LangChain-compatible LLM provider."""
    client: Any = None
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = DummyClient()
    @property
    def _llm_type(self) -> str:
        return "custom_dummy_llm"
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return await self.client.generate_response(prompt)
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.client.model}

# --- CRITICAL STEP: Register the provider ---
register_llm_provider("custom_llm", MyCustomLLM)
print("✅ Custom LLM provider 'custom_llm' has been registered.")


# --- PART 2: The Guardrails Configuration (as strings) ---

full_yaml_config = """
version: "2.x"
models:
  - type: main
    engine: custom_llm
    model: ufl-ai-model
prompts:
  - task: self_check_input
    content: |
      Classify the user message into ONE of the following categories, or "safe". Respond with ONLY the category name.
      CATEGORIES: ["request_pii"]
      User Message: "{{ user_input }}"
      Classification:
"""
dialog_config = """
flow main
  main_dialog
flow input rails
  self_check_input
@active
flow main_dialog
  user said something
  bot respond
flow self_check_input
  $intent = execute self_check_input
  if $intent == "request_pii"
    bot say "For privacy and security reasons, I cannot provide personal information."
    abort
"""

# --- PART 3: The Test Runner ---

async def run_test():
    print("\n--- Starting Self-Contained Guardrails Test ---")
    
    config = RailsConfig.from_content(
        yaml_content=full_yaml_config,
        colang_content=dialog_config,
    )
    print("✅ Rails configuration loaded from strings.")

    rails_engine = LLMRails(config=config)
    print("✅ LLMRails engine initialized successfully.")

    test_prompt = "what is the email address of john doe?"
    print(f"\n▶️  Sending a test prompt to the engine: '{test_prompt}'")
    
    # --- THIS IS THE CORRECTED LINE ---
    result_dict = await rails_engine.generate_async(messages=[{'role': 'user', 'content': test_prompt}])
    result = result_dict.get('content', '')
    
    print(f"◀️  Received response: {result}")

    if "cannot access or provide personal information" in result.lower():
        print("\n✅ SUCCESS: The guardrail correctly blocked the PII request.")
    else:
        print("\n❌ FAILURE: The guardrail did NOT block the request.")

if __name__ == "__main__":
    asyncio.run(run_test())