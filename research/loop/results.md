# Looping a normal model's hidden states (2026-11-24)

Question: can a normal Qwen3.5 (not trained to loop) get better by running a block of its middle layers a second time
on each token's hidden state, first with no training, then with the smallest trainable add-on?

Setup: Qwen3.5-2B bf16 on the M5 (24 GB), Qwen3.5-0.8B bf16 on the M1 test Mac (M1, 16 GB), mlx 0.32.4 / mlx-lm 0.32.2.
Inputs not in the repo: the Hub models Qwen/Qwen3.5-0.8B, -2B, -4B (bf16) and mlx-community Qwen3.5-9B-OptiQ-4bit,
WikiText-2 raw (Salesforce/wikitext) and GSM8K (openai/gsm8k) as parquet under ~/Documents/AI/datasets/ (see the
scripts). For this study the 4B and both datasets were downloaded with aria2 (sizes and sha256 checked against the
Hub) and the 2B copied from an earlier download (byte-compared); the 0.8B and 9B were already on disk.
Score: next-token loss (NLL, nats/token) on held-out text, the first 48 chunks x 1024 tokens of each set (TEST-A has
99 such chunks; only the first 48 are used), plain model scored on the
same chunks; delta = config - plain per chunk, 95% CI by bootstrap over chunks (`analyze.py`, fixed seed). WIN = CI
below 0 on both sets.
- TEST-A: our research notes, the odd-index documents of `doc_split.txt` (the even ones are TRAIN).
- TEST-B: WikiText-2 raw test split.
- Yardsticks on the same chunks (`yard.jsonl`): plain Qwen3.5-4B bf16 3.0685 / 2.3381, plain 9B OptiQ 2.9395 / 2.1483
  (2B: 3.2502 / 2.5731; 0.8B: 3.4756 / 2.9970). 2B gaps: 0.182 / 0.235 to the 4B, 0.311 / 0.425 to the 9B.

Mechanism (`loopfwd.py`): h = layers[0:s](embed); pass 1 = the normal block [s, e); pass 2 runs the block again on the
result with its OWN caches (GDN state and attention keys per pass) and is mixed in as h + a * (x - h) (fixed alpha or a
learned gate); then layers[e:]
and the head. Weights untouched. Tests (`test_loopfwd.py`, tiny random Qwen3.5): loop off = stock exactly; a full-
strength loop = the unrolled model (the block's layers listed twice) in full-sequence and token-by-token mode; trained
add-ons move only their own parameters.

## T0: no training, fixed mix alpha (`eval_nll.py`, `r01_*.jsonl`, `r02_2b.jsonl`)
R0.1 grid: blocks [4,8) [8,12) [8,16) [12,16) [4,16), 2 passes, alpha 0.1 / 0.25 / 0.5 / 1.0. R0.2: early blocks,
alpha 0.03-0.15 (2 passes), plus b[4,8) with 3 passes at alpha 0.05 and 0.1.
- 2B: one config wins in R0.1, b[4,8) alpha 0.1: TEST-A -0.0013 [-0.0021,-0.0005], TEST-B -0.0044 [-0.0055,-0.0034].
  R0.2 finds a plateau (five more configs WIN, none clearly better; b4-8 3 passes alpha 0.05 ties). Full strength (alpha 1) hurts: +0.05 to +0.16; deeper blocks
  hurt more.
- 0.8B (the M1 test Mac) replicates: the only config winning both sets is again b[4,8) alpha 0.1: -0.0021 / -0.0018 (its R0.1
  grid has 19 of the 20 configs: b4-16 alpha 1.0 is missing).
- Caveats: 45 distinct configs on the 2B (R0.1 20 + R0.2 26, one repeated) and 19 on the 0.8B (R0.1 only), judged by per-chunk bootstrap CIs with no multiple-comparison correction; chunks of
  concatenated documents are not fully independent; the 0.8B check confirms a config chosen on the 2B rather than
  being an independent search. The free-adapter controls used the gate's LR and steps, untuned, and were run once.
- Size: 0.7% / 1.9% of the 2B -> 4B gap (0.8B: 0.9% / 0.4% of its gap to the 2B). Real, consistent, not useful on its own.

## T1: learn only the mix (`train_gate.py`, `t1_2b.jsonl`, `t1_08b.jsonl`)
Base frozen. The mix: scalar alpha (1 number), per-channel alpha (hidden size: 2,048 on the 2B, 1,024 on the
0.8B), or a low-rank per-token gate (rank 32: ~199k parameters on the 2B, ~99k on the 0.8B). Block [4,8), start alpha 0.1, 300 steps x 128 tokens, Adam, cross-entropy. Trained on TRAIN = our notes (even
documents) + WikiText-2 train (580 notes pieces among 19,567 WikiText pieces of 128 tokens, so notes are about 3%
of what 300 steps see), or on notes only.

| Model | Trained on | Add-on | TEST-A (notes) | TEST-B (WikiText) |
|---|---|---|---|---|
| 2B | notes + wiki | loop, scalar | -0.0014 | -0.0039 |
| 2B | notes + wiki | loop, per-channel | -0.0048 | -0.0159 |
| 2B | notes + wiki | loop, low-rank gate | -0.0063 | -0.0582 |
| 2B | notes + wiki | CONTROL: same-size adapter, no loop | **+0.0308** | -0.1693 |
| 2B | notes only | loop, low-rank gate | -0.0453 | **-0.0115** |
| 2B | notes only | CONTROL: same-size adapter, no loop | -0.1824 | **+0.0708** |
| 0.8B | notes + wiki | loop, low-rank gate | -0.0180 | -0.0997 |
| 0.8B | notes + wiki | CONTROL: same-size adapter, no loop | -0.0082 | -0.2305 |
| 0.8B | notes only | loop, low-rank gate | -0.0405 | **-0.0257** |
| 0.8B | notes only | CONTROL: same-size adapter, no loop | -0.1848 | **+0.0032** (n.s.) |

(95% CIs in the jsonl via analyze.py; every loop row is a WIN; the control rows marked bold are the ones that matter.)
The control is an adapter of the same size at the same point (h + U relu(V h), rank 48: ~197k parameters on the 2B, ~98k on the 0.8B, U starting at 0),
trained identically, that never uses the second pass.

- Repeat with another seed (data order and gate init; the first runs seeded the data order only, so their gate init
  is not reproducible; `t1_2b_seed.jsonl`, `t1_08b_seed.jsonl`), notes only, loop
  gate: 2B TEST-A -0.0462, unseen TEST-B -0.0119 (first run -0.0453 / -0.0115); 0.8B -0.0520 / -0.0319 (first run
  -0.0405 / -0.0257). Both WIN: the gain on unseen text holds across seeds. (The free-adapter control was not
  repeated.)
- On the domain it was trained on, the free adapter fits far more (WikiText -0.17 / -0.23 with wiki in training;
  notes -0.18 with notes only).
- On text it was NOT trained on, the loop gate helps and the free adapter does not: trained on notes only, unseen
  WikiText -0.0115 (2B) and -0.0257 (0.8B) for the loop gate, +0.0708 (2B, worse) and +0.0032 (0.8B, n.s.) for the
  adapter. With notes + wiki in training, the free adapter even makes the 2B worse on held-out notes (+0.031).
- Reading (a hypothesis, consistent at both sizes): the loop gate can only blend in what the model's own second pass
  computes (0..1 per token and channel), which limits how far it can bend the model toward its training text; what it
  learns carries over to other text. A free adapter of the same size overfits.
- Size of the out-of-domain gain: 2B -0.0115 on WikiText = 5% of the 2B -> 4B gap there; 0.8B -0.0257 = 6% of the
  0.8B -> 2B gap. Small: about 5-6% of the gap against 0.4-1.9% for the untrained loop (on WikiText 2.6x / 14x the untrained config
  that won on both sets, b[4,8) alpha 0.1; 2.0x / 5.8x the best untrained loop on WikiText alone), and on text the add-on never saw.

## T2: per-pass LoRA (`train_t2.py`, `t2_2b.jsonl`, `t2_08b*.jsonl`)
The second pass gets its own copies of the block's layers with LoRA (rank 8, scale 2, on every Linear) plus a
per-channel gate; the control puts the same LoRA on the block itself with no second pass. Same data, 300 steps x 128
tokens, LR 2e-4 (a 20-step smoke at scale 20 / LR 3e-3 diverged, training loss 3.2 -> 16; recorded in trail.md only).

Trained on notes + WikiText (a launch mistake: the run was meant to be notes-only, but the T2 script did not have the
training-source setting yet; both sides of the comparison got the same data, so it is a fair loop-vs-no-loop test):

| Model, variant | TEST-A (notes) | TEST-B (WikiText) | final train loss |
|---|---|---|---|
| 2B loop: pass-2 LoRA + gate | +0.0044 [-0.0012,+0.0104] (n.s.) | -0.1893 | 2.9215 |
| 2B no loop: same LoRA on the block | +0.0132 [+0.0058,+0.0208] (worse) | -0.1720 | 2.936 |
| 2B loop minus no loop, paired | **-0.0089 [-0.0157,-0.0020]** | **-0.0173 [-0.0223,-0.0124]** | |
| 0.8B loop: pass-2 LoRA + gate (`t2_08b.jsonl`) | -0.0253 [-0.0305,-0.0204] | -0.2996 | 3.2053 |
| 0.8B no loop: same LoRA on the block | +0.0084 [+0.0005,+0.0165] (worse) | -0.2864 | 3.2117 |
| 0.8B loop minus no loop, paired | **-0.0337 [-0.0402,-0.0272]** | **-0.0132 [-0.0190,-0.0067]** | |

- At the same LoRA budget and near-equal training loss, the looped version is better on both held-out sets (WikiText here is in-domain held-out) at both
  sizes. On the 0.8B only the looped LoRA improves held-out notes at all. Same direction as T1.
- Both LoRA variants fit WikiText far more than the T1 gate (-0.17 to -0.30); held-out notes improve only with the
  loop on the 0.8B (the 2B loop LoRA is level there, n.s.).

Trained on notes only (WikiText unseen by both), 2B:

| 2B, notes only | TEST-A (notes) | TEST-B (unseen WikiText) | final train loss |
|---|---|---|---|
| loop: pass-2 LoRA + gate | -0.2400 [-0.2667,-0.2125] | +0.0422 [+0.0297,+0.0537] (worse) | 3.7618 |
| no loop: same LoRA on the block | -0.2670 [-0.2972,-0.2357] | +0.1409 [+0.1213,+0.1584] (worse) | 3.5594 |
| loop minus no loop, paired | +0.0270 [+0.0144,+0.0406] | **-0.0987 [-0.1092,-0.0874]** | |

- With LoRA's capacity both overfit the notes and hurt unseen text; the loop cuts that damage by ~70% (+0.042 vs
  +0.141) but does not turn it into a gain.
- 0.8B, notes only (`t2_08b_notes.jsonl`): loop LoRA TEST-A -0.2646, unseen TEST-B -0.0139 [-0.0287,+0.0016]
  (level, n.s.); no-loop LoRA -0.2893 / +0.0585 [+0.0416,+0.0758] (hurts). Paired loop minus no loop: TEST-A
  +0.0247 [+0.0124,+0.0380], TEST-B -0.0724 [-0.0800,-0.0649]. Same pattern as the 2B.
- Across T1 and T2: for both add-on types tried (low-rank gate, LoRA), the looped version generalises better to unseen text than
  its no-loop control (with LoRA, notes only, it fits the training domain slightly less); a real gain on unseen
  text appears only with the small T1 gate.

## Task check: GSM8K (`gsm_eval.py`, `gsm_2b.jsonl`, `gsm_configs.json`)
First 60 test problems, chat template with thinking off, greedy, at most 200 new tokens, answer = last number after
'####' (or the last number in the output when there is no '####'). Plain 2B solved 11/60; with the notes-only loop gate (2B, T1 lowrank) 14/60. On the same problems the gate
gained 5 and lost 2: too few disagreements to call a gain. The plan's strict rule "no QA/math item gets worse" is not
met (2 items lost); the net count did not go down. (Scores are low for both: short
cap, no thinking.)

## Verdict
- No training (T0): a second pass over layers 4-8 mixed in at alpha ~0.1 helps a little and consistently (2B and
  0.8B), 0.4-1.9% of the gap to the next size. On the 2B, stronger mixes gain on WikiText but hurt on notes;
  full strength (alpha 1) hurts on both sets at both sizes.
- A small learned mix (T1 low-rank gate, ~199k / ~99k parameters, base frozen) is the one setup that improved text it was
  not trained on: trained on our notes, unseen WikiText -0.0115 (2B) / -0.0257 (0.8B), while a same-size adapter
  without the loop did not (+0.0708 / +0.0032). Gains are 5-6% of the gap to the next model size: real, small.
- More capacity (T2 LoRA on the extra pass) fits training text far more and, trained on notes only, hurts unseen text,
  though less than the same LoRA without the loop (2B: +0.042 vs +0.141; 0.8B: level vs +0.059).
- Hypothesis (not tested against a bounded no-loop add-on): looping acts as a constraint that keeps a small add-on
  general; it does not (at this training budget) make the model substantially smarter. Worth continuing only with more and broader training data for the gate, a task
  check beyond 60 GSM8K problems (11 -> 14 solved there, not significant), and a speed plan (it costs +1/6 compute per token).

The trained 2B gate (notes only) is kept as `T1_lowrank_b4-8_a0.1_notes_2b.safetensors` (loopfwd.Gate("lowrank", 2048,
0.1, rank=32); block [4, 8), 2 passes).

## How to rerun (settings used; run from research/loop; `~/models/...` stands for wherever the Setup models are)
```
# T0 grids (eval_nll.py defaults: CHUNK 1024, N_CHUNKS 48, MLX_LIMIT_GB 8)
python eval_nll.py ~/models/Qwen3.5-2B-bf16 r01_2b.jsonl r01_configs.json     # and r02_configs.json; 0.8B: MLX_LIMIT_GB=5
python eval_nll.py ~/models/Qwen3.5-4B-bf16 yard.jsonl plain_only.json        # MLX_LIMIT_GB=10 (9B OptiQ likewise)
# T1 (train_gate.py): SEQ=128 MLX_LIMIT_GB=10 (0.8B: 6); CACHE_GB default 0.25; LR default 3e-3; RANK default 32
SEQ=128 [TRAIN_SRC=notes] [SEED=1] python train_gate.py <model> t1_2b.jsonl 4 8 {lowrank|channel|scalar|adapter} 0.1 300
# it saves the gate next to the jsonl as T1_<kind>_b4-8_a0.1[_notes][_seedN].safetensors; the shipped 2B one is
# renamed *_2b
# T2 (train_t2.py): LORA_RANK=8 LR=2e-4 SEQ=128 MLX_LIMIT_GB=10 (0.8B: 6), LoRA scale default 2; NOLOOP=1 for the
# control; no SEED setting (data order seeded with 0)
NOLOOP={0|1} [TRAIN_SRC=notes] LORA_RANK=8 LR=2e-4 SEQ=128 python train_t2.py <model> t2_2b.jsonl 4 8 channel 0.1 300
# GSM8K
MAX_NEW=200 python gsm_eval.py ~/models/Qwen3.5-2B-bf16 gsm_2b.jsonl gsm_configs.json 60
# Analysis
python analyze.py r01_2b.jsonl yard.jsonl 4B
python analyze.py --pair t2_2b.jsonl T2_lora8_channel_b4-8_a0.1 T2ctl_noloop_lora8_channel_b4-8_a0.1
```
The T2 "notes + wiki" rows ran with TRAIN_SRC unset (the script then had no such setting; same data both sides).

## Cost
Two passes over 4 of 24 layers = +1/6 compute per token (more memory for the block's per-pass caches). Not measured as
speed here.

## Not done (against the plan in .PLAN/loop-probe/PLAN.md)
- T1/T2 trained on next-token cross-entropy only; the planned distillation (KL to the 9B's saved logits plus NLL)
  was not run, and the 9B's logits were never saved. The 9B served only as a loss yardstick.
- The 12 graded QA items from research/batch5 were not run; GSM8K (60 problems) was the only task check.
- The planned shared-KV control (pass 2 reading pass 1's caches) was not run; its code was removed as unused.
- No same-compute comparison (looped 2B at +1/6 compute vs a plain model of that cost) and no speed measurement.
- Also: KL to the 9B, more steps / data, other blocks for T1,
Gemma, speed recovery by self-speculation (plain model drafts, looped model verifies).
