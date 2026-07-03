# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _get_gdn_prefill_warmup_seq_lens,
)


def test_gdn_prefill_warmup_uses_single_chunk_for_non_cutedsl() -> None:
    assert _get_gdn_prefill_warmup_seq_lens("triton") == ((FLA_CHUNK_SIZE,),)
    assert _get_gdn_prefill_warmup_seq_lens("flashinfer") == ((FLA_CHUNK_SIZE,),)


def test_gdn_prefill_warmup_includes_small_multi_seq_for_cutedsl() -> None:
    assert _get_gdn_prefill_warmup_seq_lens("cutedsl") == (
        (FLA_CHUNK_SIZE,),
        (1, 1, 1, 1, 1, 1, 1, 1),
    )
