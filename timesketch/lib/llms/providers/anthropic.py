# Copyright 2024 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Anthropic (Claude) LLM provider."""

import json
from typing import Any, Optional, Union

import requests

from timesketch.lib.llms.providers import interface
from timesketch.lib.llms.providers import manager

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_TIMEOUT = 120


class Anthropic(interface.LLMProvider):
    """Anthropic LLM provider for Timesketch.

    Uses the Anthropic Messages API to generate text with Claude models.

    Config keys:
        api_key (str): Anthropic API key.
        model (str): Model name (e.g. "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001").
        base_url (str): API base URL (default: https://api.anthropic.com).
        anthropic_version (str): API version header
            (default: "2023-06-01").
        timeout (int): Request timeout in seconds (default: 120).
    """

    NAME = "anthropic"

    def __init__(self, config: dict, **kwargs: Any):
        super().__init__(config, **kwargs)
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model")
        self.base_url = self.config.get("base_url", DEFAULT_BASE_URL).rstrip("/")
        self.anthropic_version = self.config.get(
            "anthropic_version", DEFAULT_ANTHROPIC_VERSION
        )
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)

        if not self.api_key:
            raise ValueError(
                "Anthropic provider requires an 'api_key' in its configuration."
            )
        if not self.model:
            raise ValueError(
                "Anthropic provider requires a 'model' in its configuration."
            )

    def generate(
        self, prompt: str, response_schema: Optional[dict] = None
    ) -> Union[dict, str]:
        """Generate text using the Anthropic Messages API.

        Args:
            prompt: The prompt to generate a response for.
            response_schema: An optional JSON schema for structured output.
                When provided, the model is instructed to return valid JSON
                matching the schema.

        Returns:
            The generated text, or a parsed dict if response_schema is provided.

        Raises:
            ValueError: If the request fails or the response cannot be parsed.
        """
        url = f"{self.base_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }

        # If a response schema is requested, prepend instructions to return JSON.
        effective_prompt = prompt
        if response_schema:
            schema_str = json.dumps(response_schema)
            effective_prompt = (
                f"{prompt}\n\nYou must respond with valid JSON matching this "
                f"schema: {schema_str}\nRespond ONLY with the JSON, no other text."
            )

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": effective_prompt}],
            "max_tokens": self.config.get(
                "max_output_tokens", interface.DEFAULT_MAX_OUTPUT_TOKENS
            ),
            "temperature": self.config.get(
                "temperature", interface.DEFAULT_TEMPERATURE
            ),
            "top_p": self.config.get("top_p", interface.DEFAULT_TOP_P),
            "top_k": self.config.get("top_k", interface.DEFAULT_TOP_K),
        }

        try:
            response = requests.post(
                url, headers=headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as error:
            raise ValueError(f"Request timed out: {error}") from error
        except requests.exceptions.HTTPError as error:
            raise ValueError(
                f"HTTP error from Anthropic API: "
                f"{response.status_code} {response.text}"
            ) from error
        except requests.exceptions.RequestException as error:
            raise ValueError(f"Error making request: {error}") from error

        try:
            response_data = response.json()
            # Anthropic returns content as a list of content blocks.
            content_blocks = response_data["content"]
            text_response = "".join(
                block["text"] for block in content_blocks if block["type"] == "text"
            ).strip()
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                f"Unexpected response structure from Anthropic API: "
                f"{response.json()}"
            ) from e

        if response_schema:
            try:
                return json.loads(text_response)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Error JSON parsing response: {text_response}: {error}"
                ) from error

        return text_response


manager.LLMManager.register_provider(Anthropic)
