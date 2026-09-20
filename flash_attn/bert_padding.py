"""
flash_attn/bert_padding.py
============================
A local shim satisfying verl's `from flash_attn.bert_padding import
index_first_axis, pad_input, rearrange, unpad_input` -- without needing
the real flash-attn package, which requires either a matching pre-built
wheel (none was found for this environment) or compiling CUDA kernels
from source (can take hours and isn't guaranteed to succeed on Colab's
free tier).

Second attempt at this file, and worth being honest about why: the
first version tried to re-export these 4 functions from
transformers.integrations.npu_flash_attention, on the assumption that
verl's own code importing the same 4 names from that module (as its
NPU-hardware alternative) meant they were interchangeable. That
assumption was wrong -- checked the actual current transformers source
directly and that module doesn't define any of these 4 functions at
all; verl's reference to it is likely drift against an older
transformers version. Also checked: einops (the obvious source for
`rearrange`) isn't a declared dependency here either, so it can't be
assumed present.

Given two wrong guesses at "borrow this from somewhere else," this
version stops guessing and does the thing that's actually verified
safe: every one of these 4 functions is called ONLY inside verl's
`if self.use_remove_padding:` block in dp_actor.py (confirmed by
reading that file directly, every call site, not assumed). Our launch
scripts set use_remove_padding=False. So these just need to EXIST as
importable names -- they never need to run. Rather than a "real-looking"
reimplementation that could be subtly wrong in some way that's much
harder to notice than a clean crash, each one is a loud, explicit
failure if anything unexpected ever does call it -- easy to debug,
impossible to silently get wrong.

If a real flash-attn install ever becomes worthwhile (e.g. for its
actual performance benefit, not just to satisfy an import), delete
this whole flash_attn/ folder and pip install the real package --
nothing else needs to change.
"""


def _not_needed(name: str):
    def _fn(*args, **kwargs):
        raise RuntimeError(
            f"flash_attn.bert_padding.{name}() was actually called, but this is "
            f"a stub -- it only exists to satisfy an import. This means "
            f"use_remove_padding somehow ended up True somewhere, which the "
            f"launch scripts explicitly set to False. Check the actual config "
            f"a run is using; this function was never meant to run for real."
        )
    _fn.__name__ = name
    return _fn


index_first_axis = _not_needed("index_first_axis")
pad_input = _not_needed("pad_input")
rearrange = _not_needed("rearrange")
unpad_input = _not_needed("unpad_input")

__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]