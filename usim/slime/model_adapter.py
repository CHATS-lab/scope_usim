"""Slime-specific ModelAdapter implementation using SGLang HTTP endpoints.

This adapter wraps Slime's SGLang HTTP client to provide a unified interface
for UserSimOrchestrator.
"""

import asyncio
import logging
from argparse import Namespace
from typing import Any, Dict, List, Optional

from usim.core.model_adapter import ModelAdapter

try:
    import weave

    _WEAVE_AVAILABLE = True
except ImportError:
    _WEAVE_AVAILABLE = False

logger = logging.getLogger(__name__)


class SlimeModelAdapter(ModelAdapter):
    """ModelAdapter for Slime backend using SGLang HTTP generation.

    This adapter wraps Slime's SGLang router to provide the standard
    ModelAdapter interface for UserSimOrchestrator. It uses async HTTP calls
    to SGLang for efficient generation.
    """

    def __init__(
        self,
        router_ip: str,
        router_port: int,
        tokenizer: Any,
        processor: Optional[Any] = None,
        default_sampling_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Slime model adapter.

        Args:
            router_ip: SGLang router IP address
            router_port: SGLang router port
            tokenizer: Tokenizer instance (from transformers)
            processor: Optional processor for multimodal inputs
            default_sampling_params: Default sampling parameters for generation
        """
        self.router_ip = router_ip
        self.router_port = router_port
        self._tokenizer = tokenizer
        self.processor = processor

        # Default sampling params following Slime conventions
        self.default_sampling_params = default_sampling_params or {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": -1,
            "max_new_tokens": 2048,
            "stop": [],
            "stop_token_ids": [],
            "skip_special_tokens": False,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer instance."""
        return self._tokenizer

    def apply_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Apply chat template to convert OpenAI messages to token IDs.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tokenize: Whether to tokenize the output
            add_generation_prompt: Whether to add generation prompt
            tools: Optional tool schemas

        Returns:
            List of token IDs when tokenize=True
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            kwargs["tools"] = tools

        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Synchronously generate responses.

        Args:
            messages: List of message dicts
            input_ids: Optional pre-computed input token IDs
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            List of generation results
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_async(messages, input_ids, temperature, top_p, max_tokens, **kwargs)
        )

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate responses using Slime's SGLang HTTP endpoint.

        Args:
            messages: List of message dicts
            input_ids: Optional pre-computed input token IDs
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (tools, etc.)

        Returns:
            List of generation results
        """
        # Import slime utilities
        try:
            from slime.utils.http_utils import post
        except ImportError:
            raise ImportError("slime package required for SlimeModelAdapter")

        # Compute input IDs if not provided
        tools = kwargs.pop("tools", None)
        if input_ids is None:
            input_ids = self.apply_template(messages, tokenize=True, tools=tools)

        # Get formatted prompt text for the response
        formatted_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **({"tools": tools} if tools else {}),
        )

        # Build sampling params
        sampling_params = self.default_sampling_params.copy()
        sampling_params.update({
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        })
        sampling_params.update(kwargs)

        url = f"http://{self.router_ip}:{self.router_port}/generate"

        try:
            payload = {
                "input_ids": input_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            output = await post(url, payload)

            # Extract response tokens and logprobs
            response_tokens = []
            response_logprobs = []

            if "output_token_logprobs" in output.get("meta_info", {}):
                for item in output["meta_info"]["output_token_logprobs"]:
                    response_logprobs.append(item[0])
                    response_tokens.append(item[1])

            return [{
                "prompt": formatted_prompt,
                "text": output.get("text", ""),
                "token_ids": response_tokens,
                "logprobs": response_logprobs,
                "prompt_token_ids": input_ids,
            }]

        except Exception as e:
            logger.error(f"Slime generation failed: {e}")
            return [{
                "prompt": formatted_prompt,
                "text": "",
                "token_ids": [],
                "logprobs": [],
                "prompt_token_ids": input_ids if input_ids else [],
                "error": str(e),
            }]


def create_slime_model_adapter(args: Namespace) -> SlimeModelAdapter:
    """Factory function to create SlimeModelAdapter from Slime args.

    Args:
        args: Slime Namespace with sglang_router_ip, sglang_router_port, etc.

    Returns:
        Configured SlimeModelAdapter instance
    """
    try:
        from slime.utils.processing_utils import load_tokenizer, load_processor
    except ImportError:
        raise ImportError("slime package required for create_slime_model_adapter")

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

    # Build sampling params from args
    default_sampling_params = {
        "temperature": getattr(args, "rollout_temperature", 0.7),
        "top_p": getattr(args, "rollout_top_p", 0.9),
        "top_k": getattr(args, "rollout_top_k", -1),
        "max_new_tokens": getattr(args, "rollout_max_response_len", 2048),
        "stop": getattr(args, "rollout_stop", []),
        "stop_token_ids": getattr(args, "rollout_stop_token_ids", []),
        "skip_special_tokens": getattr(args, "rollout_skip_special_tokens", False),
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    return SlimeModelAdapter(
        router_ip=args.sglang_router_ip,
        router_port=args.sglang_router_port,
        tokenizer=tokenizer,
        processor=processor,
        default_sampling_params=default_sampling_params,
    )
