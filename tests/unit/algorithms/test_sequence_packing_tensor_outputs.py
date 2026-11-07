# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""SequencePackingLossWrapper's "_"-prefixed per-token outputs over sequences of different lengths."""

import torch

from nemo_rl.algorithms.loss.wrapper import SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_per_token_outputs_are_one_padded_row_per_sequence():
    lengths = [3, 5]
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    logits = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)  # token i's "logprob" is i
    data = BatchedDataDict({"token_mask": torch.ones(2, max(lengths))})

    def prepare(logits, data, **kwargs):
        return {"next_token_logprobs": logits[..., 0]}, data

    def loss_fn(data, global_valid_seqs, global_valid_toks, next_token_logprobs):
        return next_token_logprobs.sum(), {"num_valid_samples": 1, "_curr_logprobs": next_token_logprobs}

    wrapper = SequencePackingLossWrapper(loss_fn=loss_fn, prepare_fn=prepare, cu_seqlens_q=cu_seqlens)
    _, metrics = wrapper(logits, data, global_valid_seqs=torch.tensor(2), global_valid_toks=torch.tensor(8))

    assert metrics["num_valid_samples"] == 2
    assert torch.equal(
        metrics["_curr_logprobs"],
        torch.tensor([[0.0, 1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 5.0, 6.0, 7.0]]),
    )
