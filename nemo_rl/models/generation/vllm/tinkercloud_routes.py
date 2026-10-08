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
"""TinkerCloud's routes on the vLLM worker HTTP app.

/tinkercloud/v1/generate takes the request body of vLLM's /inference/v1/generate
(token_ids, sampling_params, cache_salt) and answers with flat logprobs. vLLM's
own route builds two pydantic objects per token and serialises them with
model_dump + json.dumps; when every sequence of a step finishes together each
worker serialises its responses back to back, ~2 s per 130k tokens at any model
size. Here a choice carries token_ids, finish_reason and logprobs as
{"content": [{"logprob": x}, ...]} (the sampled token's logprob, floored to
CLAMPED_LOGPROB where vLLM has no finite value, the shape SkyRL's own generate
route uses); prompt_logprobs is a flat list with None at position 0. Encoding is
msgspec on both sides of the wire. A client disconnect cancels the handler
(with_cancellation), which aborts the engine request. The route answers from the
last RequestOutput and so forces output_kind=FINAL_ONLY, the only kind that
carries every child of an n > 1 request in it.

Kept in its own module so the fork's only in-tree touch is the registration
call in the worker's app setup.
"""

import math
from typing import Any, Optional

import msgspec
from fastapi import FastAPI, Request, Response

# Matches the floor vLLM applies at its own serving boundaries.
CLAMPED_LOGPROB = -9999.0


class GenerateRequest(msgspec.Struct):
    token_ids: list[int]
    sampling_params: dict[str, Any] = {}
    cache_salt: Optional[str] = None


class LogprobEntry(msgspec.Struct):
    logprob: float


class Logprobs(msgspec.Struct):
    content: list[LogprobEntry]


class Choice(msgspec.Struct):
    index: int
    finish_reason: Optional[str]
    token_ids: list[int]
    logprobs: Optional[Logprobs]


class GenerateResponse(msgspec.Struct):
    request_id: str
    choices: list[Choice]
    prompt_logprobs: Optional[list[Optional[float]]]


class ErrorBody(msgspec.Struct):
    error: str


def _floored(logprob: float) -> float:
    return logprob if math.isfinite(logprob) else CLAMPED_LOGPROB


def _error(status: int, message: str) -> Response:
    return Response(
        msgspec.json.encode(ErrorBody(error=message)),
        status_code=status,
        media_type="application/json",
    )


def register_tinkercloud_routes(app: FastAPI, engine_client) -> None:
    from vllm.entrypoints.utils import with_cancellation
    from vllm.inputs.data import TokensPrompt
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.utils import random_uuid

    decoder = msgspec.json.Decoder(GenerateRequest)

    @app.post("/tinkercloud/v1/generate")
    @with_cancellation
    async def generate_flat(raw_request: Request):
        try:
            req = decoder.decode(await raw_request.body())
            params = SamplingParams(**req.sampling_params)
        except (
            msgspec.DecodeError,
            msgspec.ValidationError,
            TypeError,
            ValueError,
        ) as e:
            return _error(422, f"invalid generate request: {e}")
        params.output_kind = RequestOutputKind.FINAL_ONLY
        prompt = TokensPrompt(prompt_token_ids=req.token_ids)
        if req.cache_salt is not None:
            prompt["cache_salt"] = req.cache_salt
        request_id = random_uuid()
        final = None
        try:
            async for res in engine_client.generate(
                prompt, params, request_id=request_id
            ):
                final = res
        except (
            ValueError
        ) as e:  # the engine rejected the request (e.g. prompt too long)
            return _error(400, str(e))
        if final is None:
            return _error(500, "engine returned no output")
        choices = []
        for out in final.outputs:
            ids = list(out.token_ids)
            logprobs = None
            if out.logprobs is not None:
                # the sampled token is always present in its position's dict
                logprobs = Logprobs(
                    content=[
                        LogprobEntry(logprob=_floored(out.logprobs[i][t].logprob))
                        for i, t in enumerate(ids)
                    ]
                )
            choices.append(
                Choice(
                    index=out.index,
                    finish_reason=out.finish_reason,
                    token_ids=ids,
                    logprobs=logprobs,
                )
            )
        prompt_logprobs = None
        if final.prompt_logprobs is not None:
            prompt_logprobs = [
                None if d is None else _floored(d[t].logprob)
                for d, t in zip(final.prompt_logprobs, req.token_ids)
            ]
        return Response(
            msgspec.json.encode(
                GenerateResponse(
                    request_id=request_id,
                    choices=choices,
                    prompt_logprobs=prompt_logprobs,
                )
            ),
            media_type="application/json",
        )
