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
"""OpenAI-compatible LLM provider.

Works with any API that implements the OpenAI chat completions interface,
including OpenAI, vLLM, LiteLLM, Together AI, Groq, Mistral, and others.
"""

import json
from typing import Any, Optional, Union

import requests

from timesketch.lib.llms.providers import interface
from timesketch.lib.llms.providers import manager

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 120


class OpenAICompatible(interface.LLMProvider):
    """OpenAI-compatible LLM provider.

    Supports any API server that implements the OpenAI chat completions
    endpoint (POST /chat/completions), including:
      - OpenAI (api.openai.com)
      - vLLM, LiteLLM, Ollama (OpenAI-compat mode), LocalAI
      - Together AI, Groq, Mistral, Fireworks, Anyscale, etc.

    Config keys:
        api_key (str): API key for authentication.
        model (str): Model name/ID to use (e.g. "gpt-4o", "gpt-4o-mini").
        base_url (str): Base URL of the API (default: https://api.openai.com/v1).
        timeout (int): Request timeout in seconds (default: 120).
    """

    NAME = "openai"

    def __init__(self, config: dict, **kwargs: Any):
        super().__init__(config, **kwargs)
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model")
        self.base_url = self.config.get("base_url", DEFAULT_BASE_URL).rstrip("/")
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)

        if not self.api_key:
            raise ValueError(
                "OpenAI-compatible provider requires an 'api_key' in its "
                "configuration."
            )
        if not self.model:
            raise ValueError(
                "OpenAI-compatible provider requires a 'model' in its configuration."
            )

    def generate(
        self, prompt: str, response_schema: Optional[dict] = None
    ) -> Union[dict, str]:
        """Generate text using an OpenAI-compatible chat completions API.

        Args:
            prompt: The prompt to generate a response for.
            response_schema: An optional JSON schema for structured output.

        Returns:
            The generated text, or a parsed dict if response_schema is provided.

        Raises:
            ValueError: If the request fails or the response cannot be parsed.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.get(
                "max_output_tokens", interface.DEFAULT_MAX_OUTPUT_TOKENS
            ),
            "temperature": self.config.get(
                "temperature", interface.DEFAULT_TEMPERATURE
            ),
            "top_p": self.config.get("top_p", interface.DEFAULT_TOP_P),
        }

        if response_schema:
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": response_schema,
                },
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
                f"HTTP error from OpenAI-compatible API: "
                f"{response.status_code} {response.text}"
            ) from error
        except requests.exceptions.RequestException as error:
            raise ValueError(f"Error making request: {error}") from error

        try:
            response_data = response.json()
            text_response = (
                response_data["choices"][0]["message"]["content"].strip()
            )
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected response structure: {response.json()}"
            ) from e

        if response_schema:
            try:
                return json.loads(text_response)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Error JSON parsing response: {text_response}: {error}"
                ) from error

        return text_response


manager.LLMManager.register_provider(OpenAICompatible)
