"""Feature extraction using AIDO.Protein-RAG-3B (RAGPLM) backbone via ModelGenerator.

IMPORTANT:
This model architecture (model_type 'fm4bio') is NOT yet integrated with HuggingFace
Auto(Token) classes, so attempting to load it with transformers.AutoTokenizer /
T5EncoderModel will fail (KeyError: 'fm4bio'). The official usage pattern is
through the `modelgenerator` library (https://github.com/genbio-ai/ModelGenerator).

This script uses `Embed` task with backbone `aido_protein_rag_3b` to obtain
sequence-level embeddings for proteins in the provided datasets.

If you only have single sequences (no MSA), we provide a minimal stub MSA per
sequence. The model was trained with MSAs and may perform better with real MSA
inputs; supplying only single-sequence MSAs is a degradation but still works to
extract latent representations.

USAGE (on the Linux cluster):
    pip install --upgrade "git+https://github.com/genbio-ai/ModelGenerator.git"
    # (Also ensure: pip install sentencepiece einops accelerate numpy pandas torch)
    cd feature_extraction
    python extract_features_ragplm.py

OUTPUT:
    For each dataset (SHS27k, SHS148k, STRING):
        <name>_ragplm_embeddings.npy  (float32 array shape: [N, H])
        <name>_protein_ids.txt        (protein identifiers, one per line)

We perform safe batching with per-sequence processing to avoid OOM for 3B model.
You can increase throughput by setting BATCH_MODE = True (see constant below)
if GPU memory is large (>40GB). For typical 24GB cards, per-sequence is safest.
"""
from __future__ import annotations
import os
import re
import math
import time
import json
import random
import logging
from dataclasses import dataclass
from typing import List

import torch
import numpy as np
import pandas as pd
import inspect

# ---------------- PyTorch compatibility patch ---------------- #
# Some environments have an older torch where scaled_dot_product_attention
# does not accept 'enable_gqa'. The fm4bio backbone passes this arg, so we
# inject a shim if necessary.
try:
    import torch.nn.functional as F
    if 'enable_gqa' not in inspect.signature(F.scaled_dot_product_attention).parameters:  # type: ignore[attr-defined]
        _orig_sdp = F.scaled_dot_product_attention
        def _compat_sdp(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):  # noqa: D401
            return _orig_sdp(query, key, value,
                             attn_mask=attn_mask,
                             dropout_p=dropout_p,
                             is_causal=is_causal,
                             scale=scale)
        F.scaled_dot_product_attention = _compat_sdp  # type: ignore
        print("[Compat] Patched torch.nn.functional.scaled_dot_product_attention to ignore 'enable_gqa'.")
except Exception as _patch_err:  # pragma: no cover
    print(f"[Compat] Failed to patch scaled_dot_product_attention: {_patch_err}")

try:
    from modelgenerator.tasks import Embed
except ImportError as e:
    raise SystemExit(
        "modelgenerator is required. Install with: pip install git+https://github.com/genbio-ai/ModelGenerator.git\n"
        f"Original import error: {e}" 
    )

# ---------------- Configuration ---------------- #
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed_data')
OUTPUT_DIR = os.path.dirname(__file__)
DATASETS = ["SHS27k", "SHS148k", "STRING"]
MAX_LENGTH = 12800               # Model context length
ALLOW_TRUNCATION = True          # If sequence > MAX_LENGTH, truncate tail
BATCH_MODE = False               # Set True to batch multiple sequences (experimental)
BATCH_SIZE = 2                   # Used only if BATCH_MODE is True
FP16 = True                      # Try half precision to save memory
SAVE_TOKEN_LEVEL = True          # Save token-level embeddings (variable length) per sequence
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("ragplm")

RESTYPES = 'ARNDCQEGHILKMFPSTWYV'  # Standard 20 amino acids
VALID_AA_PATTERN = re.compile(r"[^ARNDCQEGHILKMFPSTWYVXBZOU]", re.IGNORECASE)


def clean_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    # Replace uncommon / ambiguous with 'X'
    seq = re.sub(r"[UZOB]", "X", seq)
    # Remove invalid characters (keep X and standard AA)
    seq = VALID_AA_PATTERN.sub("X", seq)
    return seq


def prepare_single_msa(seq: str) -> List[str]:
    """Create a minimal MSA list for a single sequence case.
    We give the original sequence plus one lightly noised variant as weak augmentation.
    """
    if len(seq) < 10:
        return [seq]
    arr = list(seq)
    # introduce up to 1% random noise (at least 1) without changing drastically
    n_mut = max(1, len(seq)//100)
    idxs = random.sample(range(len(seq)), n_mut)
    for i in idxs:
        arr[i] = random.choice(RESTYPES)
    mutated = ''.join(arr)
    return [seq, mutated]


def mean_pool_embedding(t: torch.Tensor) -> torch.Tensor:
    """If embedding is token-level (B, L, H), mean-pool; if already (B, H) return as-is."""
    if t.ndim == 3:
        return t.mean(dim=1)
    return t


def load_model(device: torch.device):
    logger.info("Loading RAGPLM backbone via modelgenerator (aido_protein_rag_3b)...")
    model = Embed.from_config({"model.backbone": "aido_protein_rag_3b"}).eval()
    model.backbone.max_length = MAX_LENGTH
    if FP16 and device.type == 'cuda':
        model = model.to(device=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    else:
        model = model.to(device)
    logger.info("Model loaded.")
    return model


def _sanitize_id(pid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", pid)[:100]


def process_dataset(name: str, model, device: torch.device):
    input_csv = os.path.join(DATA_DIR, f"protein.{name}.sequences.dictionary.csv")
    emb_path = os.path.join(OUTPUT_DIR, f"{name}_ragplm_embeddings.npy")
    id_path = os.path.join(OUTPUT_DIR, f"{name}_protein_ids.txt")

    if os.path.exists(emb_path):
        logger.info(f"[Skip] Embeddings already exist for {name}: {emb_path}")
        return

    if not os.path.isfile(input_csv):
        logger.warning(f"File not found for {name}: {input_csv}")
        return

    df = pd.read_csv(input_csv, header=None)
    protein_ids = df[0].astype(str).tolist()
    sequences = [clean_sequence(s) for s in df[1].astype(str).tolist()]
    logger.info(f"Loaded {len(sequences)} sequences for {name}.")

    all_vecs = []
    start_time = time.time()

    token_dir = None
    if SAVE_TOKEN_LEVEL:
        token_dir = os.path.join(OUTPUT_DIR, f"{name}_token_embeddings")
        os.makedirs(token_dir, exist_ok=True)

    if BATCH_MODE:
        logger.info(f"Batch mode enabled (batch_size={BATCH_SIZE})")
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_ids = protein_ids[i:i+BATCH_SIZE]
            batch_seqs = sequences[i:i+BATCH_SIZE]
            data = {
                'sequences': [],
                'msa': [],
                'str_emb': None,  # will fill later
            }
            max_len_in_batch = 0
            cleaned_batch = []
            msas = []
            for seq in batch_seqs:
                if len(seq) > MAX_LENGTH:
                    if ALLOW_TRUNCATION:
                        seq = seq[:MAX_LENGTH]
                    else:
                        logger.warning(f"Sequence longer than {MAX_LENGTH}, skipping.")
                        continue
                cleaned_batch.append(seq)
                msas.append(prepare_single_msa(seq))
                max_len_in_batch = max(max_len_in_batch, len(seq))
            if not cleaned_batch:
                continue
            data['sequences'] = cleaned_batch
            data['msa'] = msas
            # Create structural embedding placeholder (zeros)
            data['str_emb'] = np.zeros((len(cleaned_batch), max_len_in_batch, 384), dtype=np.float32)
            transformed = model.transform(data)
            with torch.no_grad():
                out = model(transformed)  # (B, L, H)
            if SAVE_TOKEN_LEVEL and token_dir is not None:
                # Save each sequence token embedding separately
                for local_idx, seq in enumerate(cleaned_batch):
                    pid = batch_ids[local_idx]
                    token_emb = out[local_idx, :len(seq)].detach().cpu().float().numpy()
                    np.save(os.path.join(token_dir, f"{_sanitize_id(pid)}.npy"), token_emb)
            pooled = mean_pool_embedding(out)
            all_vecs.append(pooled.cpu().float())
            logger.info(f"Processed batch {i//BATCH_SIZE + 1}/{math.ceil(len(sequences)/BATCH_SIZE)}")
    else:
        for idx, (pid, seq) in enumerate(zip(protein_ids, sequences), start=1):
            if len(seq) > MAX_LENGTH:
                if ALLOW_TRUNCATION:
                    seq = seq[:MAX_LENGTH]
                else:
                    logger.warning(f"Sequence {pid} length {len(seq)} exceeds {MAX_LENGTH}; skipping.")
                    continue
            data = {
                'sequences': [seq],
                'msa': [prepare_single_msa(seq)],
                'str_emb': np.zeros((1, len(seq), 384), dtype=np.float32)
            }
            transformed = model.transform(data)
            with torch.no_grad():
                out = model(transformed)  # (1, L, H)
            if SAVE_TOKEN_LEVEL and token_dir is not None:
                token_emb = out[0, :len(seq)].detach().cpu().float().numpy()
                np.save(os.path.join(token_dir, f"{_sanitize_id(pid)}.npy"), token_emb)
            pooled = mean_pool_embedding(out)  # shape (1, H)
            all_vecs.append(pooled.cpu().float())
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  {idx}/{len(sequences)} sequences done ({elapsed/60:.1f} min)")

    if not all_vecs:
        logger.warning(f"No embeddings produced for {name}")
        return

    emb_tensor = torch.cat(all_vecs, dim=0)
    np.save(emb_path, emb_tensor.numpy())
    with open(id_path, 'w') as f:
        for pid in protein_ids:
            f.write(pid + "\n")
    logger.info(f"Saved embeddings: {emb_path} (shape={tuple(emb_tensor.shape)})")
    logger.info(f"Saved ids: {id_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = load_model(device)
    for ds in DATASETS:
        process_dataset(ds, model, device)
    logger.info("All datasets processed.")


if __name__ == '__main__':
    main()
