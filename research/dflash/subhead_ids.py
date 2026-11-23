"""Draft head over an arbitrary id set (e.g. ids < 32768 plus the prompt's ids), for dflash_child.py's VOCAB_PROMPT.

The draft scores only the rows in `idx`; scatter_logits puts them back at their own ids in a full-vocab row with -1e30
elsewhere, so argmax, masks and token ids keep their meaning. The target still verifies with the full head.
"""
import mlx.core as mx

from mlx_fastpath.draft import sub_logits


def rows_head(head, idx):
    """The LM head's output rows `idx` (sorted int32 array), in mlx_fastpath.draft's sub-head format."""
    if hasattr(head, "scales"):
        return ("q", head.weight[idx], head.scales[idx], head.biases[idx], head.group_size, head.bits)
    return ("f", head.weight[idx])


def scatter_logits(sub_out, idx, vocab):
    full = mx.full(sub_out.shape[:-1] + (vocab,), -1e30, dtype=sub_out.dtype)
    return mx.put_along_axis(full, mx.broadcast_to(idx, sub_out.shape), sub_out, axis=-1)


def draft_ids(prompt_ids, k):
    """ids < k plus every prompt id, sorted."""
    return mx.array(sorted(set(range(k)) | {int(t) for t in prompt_ids}), dtype=mx.int32)


def subset_logits(head, idx, h, vocab):
    return scatter_logits(sub_logits(head, h), idx, vocab)
