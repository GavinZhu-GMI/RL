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
"""VllmAsyncGenerationWorker.generate_async without an engine: per-row
SamplingParams from `_tinker_*` columns, prompt logprobs for rows that ask,
and abort of the vLLM request when the worker task is cancelled."""
import asyncio

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm import vllm_generation
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorker,
)

PAD = 0


class SamplingParamsStub:
    def __init__(self, **kw):
        self.prompt_logprobs = None
        self.__dict__.update(kw)


class _Logprob:
    def __init__(self, lp):
        self.logprob = lp


class _Completion:
    def __init__(self, token_ids, logprobs):
        self.token_ids = token_ids
        self.logprobs = logprobs


class _RequestOutput:
    def __init__(self, gen_ids, prompt_ids, with_prompt_logprobs):
        self.outputs = [_Completion(gen_ids, [{t: _Logprob(-0.25)} for t in gen_ids])]
        self.prompt_logprobs = (
            [None] + [{t: _Logprob(-1.0 - i)} for i, t in enumerate(prompt_ids[1:])]
            if with_prompt_logprobs
            else None
        )


class FakeAsyncLLM:
    def __init__(self, hang=False):
        self.hang = hang
        self.requests = []
        self.aborted = []

    async def generate(self, prompt, sampling_params, request_id):
        self.requests.append(sampling_params)
        try:
            if self.hang:
                await asyncio.sleep(3600)
            yield _RequestOutput(
                [11, 12], prompt["prompt_token_ids"], sampling_params.prompt_logprobs is not None
            )
        except (asyncio.CancelledError, GeneratorExit):
            self.aborted.append(request_id)  # what vllm's AsyncLLM.generate does
            raise


def make_worker(llm, max_model_len=20):
    worker = object.__new__(VllmAsyncGenerationWorker.__ray_metadata__.modified_class)
    worker.cfg = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "_pad_token_id": PAD,
        "vllm_cfg": {"async_engine": True, "max_model_len": max_model_len},
    }
    worker.SamplingParams = SamplingParamsStub
    worker.llm = llm
    return worker


def one_row(prompt, **columns):
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([prompt + [PAD] * 2]),
            "input_lengths": torch.tensor([len(prompt)]),
        }
    )
    for k, v in columns.items():
        data[k] = [v]
    return data


async def collect(worker, data):
    results = []
    async for idx, batch in worker.generate_async(data):
        results.append((idx, batch))
    return results


def test_per_row_columns_build_the_request_capped_by_context():
    llm = FakeAsyncLLM()
    data = one_row(
        [1, 2, 3, 4, 5],
        _tinker_max_new_tokens=50,
        _tinker_temperature=0.3,
        _tinker_top_p=0.5,
        _tinker_top_k=5,
        _tinker_seed=9,
        _tinker_stop_token_ids=[7],
        _tinker_prompt_logprobs=False,
        stop_strings=["x"],
    )
    results = asyncio.run(collect(make_worker(llm), data))
    (sp,) = llm.requests
    assert sp.max_tokens == 15  # min(50, max_model_len 20 - prompt 5)
    assert (sp.temperature, sp.top_p, sp.top_k, sp.seed) == (0.3, 0.5, 5, 9)
    assert sp.stop_token_ids == [7] and sp.stop == ["x"]
    assert sp.prompt_logprobs is None
    (idx, out), = results
    assert idx == 0
    assert out["output_ids"][0].tolist() == [1, 2, 3, 4, 5, 11, 12]
    assert out["generation_lengths"].tolist() == [2]
    assert out["logprobs"][0, 5:].tolist() == pytest.approx([-0.25, -0.25])
    assert out["logprobs"][0, :5].tolist() == [0.0] * 5


def test_without_per_row_columns_config_defaults_apply():
    llm = FakeAsyncLLM()
    asyncio.run(collect(make_worker(llm), one_row([1, 2, 3])))
    (sp,) = llm.requests
    assert sp.max_tokens == 17 and sp.temperature == 0.7 and sp.seed is None


def test_prompt_logprobs_land_on_prompt_positions():
    llm = FakeAsyncLLM()
    data = one_row([1, 2, 3, 4], _tinker_prompt_logprobs=True)
    ((_, out),) = asyncio.run(collect(make_worker(llm), data))
    assert llm.requests[0].prompt_logprobs == 1
    assert out["logprobs"][0, :4].tolist() == pytest.approx([0.0, -1.0, -2.0, -3.0])


def test_cancelling_the_worker_task_aborts_the_vllm_request():
    async def run():
        llm = FakeAsyncLLM(hang=True)
        task = asyncio.create_task(collect(make_worker(llm), one_row([1, 2, 3])))
        while not llm.requests:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(llm.aborted) == 1

    asyncio.run(run())


def two_rows(**columns):
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, PAD], [4, 5, PAD, PAD]]),
            "input_lengths": torch.tensor([3, 2]),
        }
    )
    for k, v in columns.items():
        data[k] = v
    return data


def test_generate_rows_async_returns_every_row_and_aborts_all_on_cancel():
    async def run():
        llm = FakeAsyncLLM()
        rows = await make_worker(llm).generate_rows_async(two_rows(_tinker_seed=[3, 4]))
        assert sorted(idx for idx, _ in rows) == [0, 1]
        assert sorted(sp.seed for sp in llm.requests) == [3, 4]
        by_idx = dict(rows)
        assert by_idx[0]["output_ids"][0].tolist() == [1, 2, 3, 11, 12]
        assert by_idx[1]["output_ids"][0].tolist() == [4, 5, 11, 12]

        llm = FakeAsyncLLM(hang=True)
        task = asyncio.create_task(make_worker(llm).generate_rows_async(two_rows()))
        while len(llm.requests) < 2:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(llm.aborted) == 2

    asyncio.run(run())


class _FakeRef:
    """What run_single_worker_single_data hands back: awaitable via future()."""

    def __init__(self):
        import concurrent.futures

        self.fut = concurrent.futures.Future()

    def future(self):
        return self.fut


class _FakeWorkerGroup:
    dp_size = 2

    def __init__(self):
        self.calls = []
        self.refs = []

    def get_dp_leader_worker_idx(self, shard_idx):
        return shard_idx * 10

    def run_single_worker_single_data(self, method_name, worker_idx, **kwargs):
        self.calls.append((method_name, worker_idx, kwargs))
        ref = _FakeRef()
        self.refs.append(ref)
        return ref


def make_driver(worker_group):
    driver = object.__new__(VllmGeneration)
    driver.cfg = {"vllm_cfg": {"async_engine": True}}
    driver.worker_group = worker_group
    driver.current_generate_dp_shard_idx = 0
    return driver


def test_driver_rows_async_round_robins_and_cancels_the_worker_task(monkeypatch):
    cancelled = []
    monkeypatch.setattr(vllm_generation.ray, "cancel", lambda ref, force: cancelled.append((ref, force)))

    async def run():
        wg = _FakeWorkerGroup()
        driver = make_driver(wg)
        data = one_row([1, 2, 3])
        task = asyncio.create_task(driver.generate_rows_async(data))
        await asyncio.sleep(0.01)
        wg.refs[0].fut.set_result([(0, BatchedDataDict({"generation_lengths": torch.tensor([2])}))])
        [(idx, out)] = await task
        assert idx == 0 and out["gen_leader_worker_idx"] == [0]
        assert wg.calls[0][0] == "generate_rows_async" and wg.calls[0][1] == 0
        assert wg.calls[0][2]["data"] is data

        task = asyncio.create_task(driver.generate_rows_async(data))
        await asyncio.sleep(0.01)
        assert wg.calls[1][1] == 10  # next DP leader
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled == [(wg.refs[1], False)]

    asyncio.run(run())
