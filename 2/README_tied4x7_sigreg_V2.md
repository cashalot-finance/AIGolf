# Tied 4x7 + SIGReg variant on top of PR315 — v2 fixes

This revision keeps the same high-level idea (4 unique blocks, 7 effective passes, low-rank gated FFN, SIGReg auxiliary loss), but fixes three important implementation issues.

## What changed

### 1) SIGReg now samples **bottleneck** latents, not pre-head latents

The latent sample is now taken **after `middle_repeats` and before the decoder path**:

- old behavior: sample at the end of `_run_trunk`, right before `final_norm` / LM head
- new behavior: sample at the bottleneck, so the decoder can reconstruct anisotropic token features back into text

This is the safer placement for a Gaussianizing regularizer in a language model.

### 2) XSA is now applied on the **last effective layers**, not the last unique blocks

With weight tying, `XSA_LAST_N=4` and `NUM_LAYERS=4` previously enabled XSA on **all** 4 unique blocks, which meant XSA leaked into all 28 effective layers after unrolling.

Now XSA is selected **per effective depth inside `_run_trunk`**:

- old behavior: static flag on shared blocks during init
- new behavior: `depth_idx >= effective_layers - XSA_LAST_N`

So with `NUM_LAYERS=4`, `LAYER_REPEATS=7`, `XSA_LAST_N=4`, XSA only hits the last 4 effective layers of the unrolled stack.

### 3) The intended 1024-context regime is now explicit

The launch script now sets:

- `TRAIN_SEQ_LEN=1024`
- `EVAL_SEQ_LEN=1024`

The training script defaults were also aligned to the intended variant:

- `TRAIN_BATCH_TOKENS=393216`
- `TRAIN_SEQ_LEN=1024`
- `EVAL_SEQ_LEN=1024`
- `MLP_MULT=1.5`

Additionally, RoPE now receives the actual `train_seq_len` instead of a hardcoded `1024` inside attention.

## Files

- `train_gpt_tied4x7_sigreg_v2.py` — corrected training / export / eval script
- `run_tied4x7_sigreg_v2.sh` — corrected launch script
- `fixes_tied4x7_sigreg_v2.diff` — unified diff vs the previous script

## Sanity notes

The new script keeps the original choices that were *not* part of the bug report:

- low-rank gated FFN
- shared-depth encoder / bottleneck / decoder unroll
- BigramHash
- SmearGate
- partial RoPE
- EMA / mixed int6 export / sliding-window eval

The main behavioral change is that the two risky mechanisms are now placed where they were intended:

- SIGReg at the bottleneck
- XSA only on the effective tail
