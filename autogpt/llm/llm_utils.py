from __future__ import annotations

import functools
import time
from itertools import islice
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import openai
import openai.api_resources.abstract.engine_api_resource as engine_api_resource
import openai.util
import tiktoken
from colorama import Fore, Style
from openai.error import APIError, RateLimitError
from openai.openai_object import OpenAIObject

from autogpt.config import Config
from autogpt.llm.api_manager import ApiManager
from autogpt.llm.base import Message
from autogpt.llm.providers.openai import OPEN_AI_EMBEDDING_MODELS
from autogpt.logs import logger


def metered(func):
    """Adds ApiManager metering to functions which make OpenAI API calls"""
    api_manager = ApiManager()

    openai_obj_processor = openai.util.convert_to_openai_object

    def update_usage_with_response(response: OpenAIObject):
        try:
            usage = response.usage
            logger.debug(f"Reported usage from call to model {response.model}: {usage}")
            api_manager.update_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens if "completion_tokens" in usage else 0,
                response.model,
            )
        except Exception as err:
            logger.warn(f"Failed to update API costs: {err.__class__.__name__}: {err}")

    def metering_wrapper(*args, **kwargs):
        openai_obj = openai_obj_processor(*args, **kwargs)
        if isinstance(openai_obj, OpenAIObject) and "usage" in openai_obj:
            update_usage_with_response(openai_obj)
        return openai_obj

    def metered_func(*args, **kwargs):
        with patch.object(
            engine_api_resource.util,
            "convert_to_openai_object",
            side_effect=metering_wrapper,
        ):
            return func(*args, **kwargs)

    return metered_func


def retry_openai_api(
    num_retries: int = 10,
    backoff_base: float = 2.0,
    warn_user: bool = True,
):
    """Retry an OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """
    retry_limit_msg = f"{Fore.RED}Error: " f"Reached rate limit, passing...{Fore.RESET}"
    api_key_error_msg = (
        f"Please double check that you have setup a "
        f"{Fore.CYAN + Style.BRIGHT}PAID{Style.RESET_ALL} OpenAI API Account. You can "
        f"read more here: {Fore.CYAN}https://docs.agpt.co/setup/#getting-an-api-key{Fore.RESET}"
    )
    backoff_msg = (
        f"{Fore.RED}Error: API Bad gateway. Waiting {{backoff}} seconds...{Fore.RESET}"
    )

    def _wrapper(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            user_warned = not warn_user
            num_attempts = num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except RateLimitError:
                    if attempt == num_attempts:
                        raise

                    logger.debug(retry_limit_msg)
                    if not user_warned:
                        logger.double_check(api_key_error_msg)
                        user_warned = True

                except APIError as e:
                    if (e.http_status != 502) or (attempt == num_attempts):
                        raise

                backoff = backoff_base ** (attempt + 2)
                logger.debug(backoff_msg.format(backoff=backoff))
                time.sleep(backoff)

        return _wrapped

    return _wrapper


def call_ai_function(
    function: str, args: list, description: str, model: str | None = None
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    cfg = Config()
    if model is None:
        model = cfg.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args: str = ", ".join(args)
    messages: List[Message] = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}"
            f"\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    return create_chat_completion(model=model, messages=messages, temperature=0)


@metered
@retry_openai_api()
def create_text_completion(
    prompt: str,
    model: Optional[str],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
) -> str:
    cfg = Config()
    if model is None:
        model = cfg.fast_llm_model
    if temperature is None:
        temperature = cfg.temperature

    if cfg.use_azure:
        kwargs = {"deployment_id": cfg.get_azure_deployment_id_for_model(model)}
    else:
        kwargs = {"model": model}

    response = openai.Completion.create(
        **kwargs,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_output_tokens,
        api_key=cfg.openai_api_key,
    )
    return response.choices[0].text


# Overly simple abstraction until we create something better
# simple retry mechanism when getting a rate error or a bad gateway
@metered
@retry_openai_api()
def create_chat_completion(
    messages: List[Message],  # type: ignore
    model: Optional[str] = None,
    temperature: float = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (List[Message]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    cfg = Config()
    if temperature is None:
        temperature = cfg.temperature

    logger.debug(
        f"{Fore.GREEN}Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}{Fore.RESET}"
    )
    for plugin in cfg.plugins:
        if plugin.can_handle_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            message = plugin.handle_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if message is not None:
                return message
    api_manager = ApiManager()
    response = None

    if cfg.use_azure:
        kwargs = {"deployment_id": cfg.get_azure_deployment_id_for_model(model)}
    else:
        kwargs = {"model": model}

    response = api_manager.create_chat_completion(
        **kwargs,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    resp = response.choices[0].message["content"]
    for plugin in cfg.plugins:
        if not plugin.can_handle_on_response():
            continue
        resp = plugin.on_response(resp)
    return resp


def batch(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def chunked_tokens(text: str, for_model: str, chunk_length: int):
    tokenizer = tiktoken.encoding_for_model(for_model)
    tokens = tokenizer.encode(text)
    chunks_iterator = batch(tokens, chunk_length)
    yield from chunks_iterator


def get_ada_embedding(text: str) -> List[float]:
    """Get an embedding from the ada model.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding.
    """
    cfg = Config()
    model = cfg.embedding_model
    text = text.replace("\n", " ")

    if cfg.use_azure:
        kwargs = {"engine": cfg.get_azure_deployment_id_for_model(model)}
    else:
        kwargs = {"model": model}

    embedding = create_embedding(text, **kwargs)
    return embedding


@retry_openai_api()
def create_embedding(
    text: str,
    *_,
    **kwargs,
) -> openai.Embedding:
    """Create an embedding using the OpenAI API

    Args:
        text (str): The text to embed.
        kwargs: Other arguments to pass to the OpenAI API embedding creation call.

    Returns:
        openai.Embedding: The embedding object.
    """
    cfg = Config()
    token_limit = OPEN_AI_EMBEDDING_MODELS[cfg.embedding_model].max_tokens

    chunk_embeddings = []
    chunk_lengths = []
    for chunk in chunked_tokens(
        text,
        for_model=cfg.embedding_model,
        chunk_length=token_limit,
    ):
        embedding = openai.Embedding.create(
            input=[chunk],
            api_key=cfg.openai_api_key,
            **kwargs,
        )
        api_manager = ApiManager()
        api_manager.update_cost(
            prompt_tokens=embedding.usage.prompt_tokens,
            completion_tokens=0,
            model=cfg.embedding_model,
        )
        chunk_embeddings.append(embedding["data"][0]["embedding"])
        chunk_lengths.append(len(chunk))

    # do weighted avg
    chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lengths)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(
        chunk_embeddings
    )  # normalize the length to one
    chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings
