"""Protocol for model adapters across different RL backends.

This defines the interface that all framework-specific adapters must implement
to work with UserSimOrchestrator. Each backend (Slime, Tinker) provides its
own adapter that translates between the backend's model interface and this protocol.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol defining the interface for generating text with a model.

    Each RL backend implements this protocol to provide a unified interface
    for the UserSimOrchestrator.

    The adapter abstracts framework-specific operations:
    1. Template application: Converting OpenAI messages to framework-specific format
    2. Generation: Calling the framework's LLM interface
    3. Token tracking: Providing tokenizer for incremental token-in-token-out

    Implementations must support both synchronous and asynchronous generation.
    """

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer used by this model.

        Required for incremental token-in-token-out tracking during rollout.
        The tokenizer must implement apply_chat_template() and encode() methods.

        Returns:
            Tokenizer object (typically HuggingFace PreTrainedTokenizerBase)
        """
        ...

    def apply_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Apply chat template to convert OpenAI messages to token IDs.

        This abstracts framework-specific template application. Each backend
        can implement its own template system while orchestrator uses standard
        OpenAI message format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys,
                     e.g., [{'role': 'user', 'content': 'Hello'}]
            tokenize: Whether to tokenize the output into token IDs
            add_generation_prompt: Whether to add generation prompt at the end
            tools: Optional tool schemas for tool-use templates

        Returns:
            List of token IDs ready for generation
        """
        ...

    def generate(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Synchronously generate responses for prompts.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            input_ids: Optional pre-computed input token IDs
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific parameters

        Returns:
            List of generation results. Each result is a dict with:
                - 'text': Generated text (str)
                - 'token_ids': List of token IDs (List[int])
                - 'logprobs': List of log probabilities (List[float])
                - 'prompt_token_ids': Token IDs of the prompt (List[int])
        """
        ...

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Asynchronously generate responses for prompts.

        Same interface as generate() but async for better performance.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            input_ids: Optional pre-computed input token IDs
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific parameters

        Returns:
            List of generation results (same format as generate())
        """
        ...
