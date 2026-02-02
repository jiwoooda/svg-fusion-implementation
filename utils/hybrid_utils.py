"""
Hybrid SVG token utilities.

Convert model outputs to hybrid SVG token tensors and save as SVG files.
"""

import torch
from typing import Optional

from config import (
    ELEMENT_TYPES, NUM_CONTINUOUS_PARAMS,
    BOS_IDX, EOS_IDX, PAD_IDX, DEFAULT_PARAM_VAL
)


def build_special_rows(converter=None, num_continuous_params=NUM_CONTINUOUS_PARAMS):
    """
    Build BOS/EOS/PAD rows (shape [2+P]).
    """
    P = num_continuous_params
    if converter is not None:
        P = converter.num_continuous_params

    num_cols = 2 + P

    bos_row = torch.full((num_cols,), DEFAULT_PARAM_VAL, dtype=torch.long)
    bos_row[0] = BOS_IDX

    eos_row = torch.full((num_cols,), DEFAULT_PARAM_VAL, dtype=torch.long)
    eos_row[0] = EOS_IDX

    pad_row = torch.full((num_cols,), DEFAULT_PARAM_VAL, dtype=torch.long)
    pad_row[0] = PAD_IDX

    return bos_row, eos_row, pad_row


def pad_or_truncate(seq, max_len, bos_row=None, eos_row=None, pad_row=None):
    """
    Pad or truncate a [L, 2+P] sequence to max_len.
    """
    if eos_row is None or pad_row is None:
        _, eos_row, pad_row = build_special_rows()

    L = seq.shape[0]

    if L > max_len:
        seq = seq[:max_len].clone()
        seq[-1] = eos_row
    elif L < max_len:
        pad_count = max_len - L
        pads = pad_row.unsqueeze(0).expand(pad_count, -1).clone()
        seq = torch.cat([seq, pads], dim=0)

    return seq


def decode_model_outputs_to_hybrid(elem_logits, cmd_logits, cont_params_pred):
    """
    Convert model logits to a hybrid SVG token tensor.

    Args:
        elem_logits: [B, L, num_element_types]
        cmd_logits: [B, L, num_command_types]
        cont_params_pred: [B, L, P, N_BINS]

    Returns:
        [B, L, 2+P] int64 tensor
    """
    assert elem_logits.dim() == 3, f"elem_logits must be 3D, got {elem_logits.dim()}D"
    assert cmd_logits.dim() == 3, f"cmd_logits must be 3D, got {cmd_logits.dim()}D"
    assert cont_params_pred.dim() == 4, f"cont_params_pred must be 4D, got {cont_params_pred.dim()}D"

    B, L = elem_logits.shape[:2]
    P = cont_params_pred.shape[2]

    pred_elem_ids = elem_logits.argmax(dim=-1)
    pred_cmd_ids = cmd_logits.argmax(dim=-1)
    pred_bin_indices = cont_params_pred.argmax(dim=-1)

    reconstructed = torch.cat(
        [
            pred_elem_ids.unsqueeze(-1),
            pred_cmd_ids.unsqueeze(-1),
            pred_bin_indices,
        ],
        dim=-1,
    )

    assert reconstructed.shape == (B, L, 2 + P), \
        f"Expected shape ({B}, {L}, {2 + P}), got {reconstructed.shape}"

    return reconstructed.long()


def compute_actual_len(hybrid_seq, eos_idx=EOS_IDX, pad_idx=PAD_IDX):
    """
    Compute content length until the first EOS/PAD token.
    """
    elem_ids = hybrid_seq[:, 0]
    for i in range(len(elem_ids)):
        eid = int(elem_ids[i].item())
        if eid in (eos_idx, pad_idx):
            return i
    return len(elem_ids)


def save_svg_from_hybrid(hybrid_seq, converter, out_path, actual_len=None):
    """
    Save a hybrid SVG tensor to an SVG file using the provided converter.
    """
    if hybrid_seq.is_cuda:
        hybrid_seq = hybrid_seq.cpu()

    if actual_len is None:
        actual_len = compute_actual_len(hybrid_seq)

    converter.tensor_to_svg_file(
        tensor=hybrid_seq,
        output_file=str(out_path),
        actual_len=actual_len,
    )
