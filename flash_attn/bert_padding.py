"""
flash_attn/bert_padding.py
============================
A local shim satisfying verl's `from flash_attn.bert_padding import
index_first_axis, pad_input, rearrange, unpad_input` -- without needing
the real flash-attn package, which requires either a matching pre-built
wheel (none was found for this environment) or compiling CUDA kernels
from source (can take hours and isn't guaranteed to succeed on Colab's
free tier).

Why this is safe, not just convenient -- verified directly against
verl's own source before writing this, not assumed:

1. Every call site for these 4 functions inside verl/workers/actor/
   dp_actor.py sits inside one `if self.use_remove_padding:` block.
   Our launch scripts set use_remove_padding=False, so none of them
   are ever actually invoked there.
2. The other call sites (verl/utils/torch_functional.py) are either
   wrapped in their own try/except ImportError with a working fallback
   already, or are inside "rmpad"-named helper functions only ever
   called from the same padding-gated paths.
3. So technically, non-functional stubs would already be safe here.
   This goes a step further anyway: rather than stub these out, it
   re-exports transformers' own real implementation of the same 4
   functions (shipped for NPU hardware support) -- verl's own
   dp_actor.py imports these exact same 4 names from that exact
   module as its NPU-path alternative to flash_attn, which is direct
   evidence the two are meant to be interchangeable with the same
   call signature. So if anything unexpected does call these, they
   work correctly, rather than silently producing nonsense.

If a real flash-attn install ever becomes worthwhile (e.g. for its
actual performance benefit, not just to satisfy an import), delete
this whole flash_attn/ folder and pip install the real package --
nothing else needs to change.
"""

from transformers.integrations.npu_flash_attention import (
    index_first_axis,
    pad_input,
    rearrange,
    unpad_input,
)

__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]