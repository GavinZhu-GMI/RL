# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""The worker HTTP app exposes vLLM's /inference/v1/generate: the request's
SamplingParams reach ServingTokens intact and its response is returned as-is."""
import pytest

pytest.importorskip("vllm")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from vllm.entrypoints.openai.protocol import (  # noqa: E402
    ErrorResponse,
    GenerateResponse,
    GenerateResponseChoice,
)

from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    register_generate_tokens_route,
)


class FakeServingTokens:
    def __init__(self):
        self.requests = []

    async def serve_tokens(self, request, raw_request):
        self.requests.append(request)
        if request.token_ids == [404]:   # a request the engine rejects (schema-valid)
            return ErrorResponse(
                error={"message": "unknown model", "type": "NotFoundError", "code": 404}
            )
        return GenerateResponse(
            request_id="r1",
            choices=[
                GenerateResponseChoice(index=0, logprobs=None, finish_reason="length", token_ids=[7, 8])
            ],
            prompt_logprobs=None,
        )


def make_client():
    app = FastAPI()
    fake = FakeServingTokens()
    register_generate_tokens_route(app, fake)
    return TestClient(app), fake


def test_generate_forwards_sampling_params_and_returns_token_ids():
    client, fake = make_client()
    body = {
        "token_ids": [1, 2, 3],
        "sampling_params": {
            "n": 2,
            "seed": 11,
            "max_tokens": 5,
            "temperature": 0.7,
            "stop_token_ids": [9],
            "logprobs": 1,
            "detokenize": False,
        },
    }
    r = client.post("/inference/v1/generate", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["token_ids"] == [7, 8]
    (req,) = fake.requests
    sp = req.sampling_params
    assert req.token_ids == [1, 2, 3]
    assert (sp.n, sp.seed, sp.max_tokens, sp.temperature) == (2, 11, 5, 0.7)
    assert sp.stop_token_ids == [9] and sp.logprobs == 1 and sp.detokenize is False


def test_error_response_keeps_its_status_code():
    client, _ = make_client()
    r = client.post(
        "/inference/v1/generate",
        json={"token_ids": [404], "sampling_params": {"max_tokens": 1}},
    )
    assert r.status_code == 404
    assert r.json()["error"]["message"] == "unknown model"


def test_schema_invalid_request_is_rejected_before_the_engine():
    client, fake = make_client()
    r = client.post(
        "/inference/v1/generate",
        json={"token_ids": [1], "sampling_params": {"max_tokens": 0}},
    )
    assert r.status_code == 422 and fake.requests == []
