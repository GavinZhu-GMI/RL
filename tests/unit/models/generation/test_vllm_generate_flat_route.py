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
"""/tinkercloud/v1/generate answers with flat per-token logprobs: the request's
SamplingParams and cache_salt reach the engine intact, output_kind is forced to
FINAL_ONLY, each choice carries {"content": [{"logprob": x}]} with non-finite
values floored to -9999, prompt logprobs are a flat list with None at 0."""

import math

import pytest

pytest.importorskip("vllm")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from vllm.logprobs import Logprob  # noqa: E402
from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402

from nemo_rl.models.generation.vllm.tinkercloud_routes import (  # noqa: E402
    CLAMPED_LOGPROB,
    register_tinkercloud_routes,
)


class FakeEngine:
    def __init__(self):
        self.calls = []

    async def generate(self, prompt, params, request_id):
        self.calls.append((prompt, params))
        if prompt["prompt_token_ids"] == [999]:
            raise ValueError("prompt too long")
        outputs = [
            CompletionOutput(
                index=i,
                text="",
                token_ids=[10 + i, 11],
                cumulative_logprob=None,
                logprobs=[
                    {10 + i: Logprob(-0.5 - i), 3: Logprob(-0.1)},
                    {11: Logprob(-math.inf)},
                ],
                finish_reason="length",
            )
            for i in reversed(range(params.n))
        ]
        prompt_logprobs = None
        if params.prompt_logprobs is not None:
            prompt_logprobs = [None] + [
                {t: Logprob(-1.0 - i), 5: Logprob(-9.0)}
                for i, t in enumerate(prompt["prompt_token_ids"][1:])
            ]
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=prompt["prompt_token_ids"],
            prompt_logprobs=prompt_logprobs,
            outputs=outputs,
            finished=True,
        )


def make_client():
    app = FastAPI()
    engine = FakeEngine()
    register_tinkercloud_routes(app, engine)
    return TestClient(app), engine


def test_flat_logprobs_per_choice_and_params_reach_the_engine():
    client, engine = make_client()
    body = {
        "token_ids": [1, 2, 3],
        "cache_salt": "m@3",
        "sampling_params": {
            "n": 2,
            "seed": 11,
            "max_tokens": 5,
            "temperature": 0.7,
            "stop_token_ids": [9],
            "logprobs": 0,
            "detokenize": False,
        },
    }
    r = client.post("/tinkercloud/v1/generate", json=body)
    assert r.status_code == 200, r.text
    choices = sorted(r.json()["choices"], key=lambda c: c["index"])
    assert [c["token_ids"] for c in choices] == [[10, 11], [11, 11]]
    flat = [[e["logprob"] for e in c["logprobs"]["content"]] for c in choices]
    assert flat == [[-0.5, CLAMPED_LOGPROB], [-1.5, CLAMPED_LOGPROB]]  # -inf -> floor
    assert choices[0]["finish_reason"] == "length"
    assert r.json()["prompt_logprobs"] is None
    ((prompt, sp),) = engine.calls
    assert prompt["prompt_token_ids"] == [1, 2, 3] and prompt["cache_salt"] == "m@3"
    assert (sp.n, sp.seed, sp.max_tokens, sp.temperature) == (2, 11, 5, 0.7)
    assert sp.stop_token_ids == [9] and sp.logprobs == 0 and sp.detokenize is False
    assert sp.output_kind is RequestOutputKind.FINAL_ONLY  # forced by the route


def test_prompt_logprobs_are_flat_with_none_at_position_zero():
    client, _ = make_client()
    body = {
        "token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 1, "prompt_logprobs": 1, "detokenize": False},
    }
    r = client.post("/tinkercloud/v1/generate", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["prompt_logprobs"] == [None, -1.0, -2.0]


def test_invalid_requests_are_rejected_before_the_engine_and_engine_errors_are_400():
    client, engine = make_client()
    r = client.post(
        "/tinkercloud/v1/generate",
        json={"token_ids": [1], "sampling_params": {"max_tokens": 0}},
    )
    assert r.status_code == 422 and engine.calls == []
    r = client.post("/tinkercloud/v1/generate", json={"sampling_params": {}})  # schema
    assert r.status_code == 422 and engine.calls == []
    r = client.post(
        "/tinkercloud/v1/generate",
        json={"token_ids": [999], "sampling_params": {"max_tokens": 1}},
    )
    assert r.status_code == 400 and "prompt too long" in r.json()["error"]
