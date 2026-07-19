# audiogear

A configurable pipeline for **preparing and annotating speech datasets for TTS**.
Given a folder of audio (with or without transcripts), audiogear computes rich
per-clip features — speech quality, prosody, intelligibility, speaker, and
transcription — so you can filter, balance, and describe a dataset the way
[DataSpeech](https://github.com/huggingface/dataspeech) does.

It is **language-agnostic**: every model is chosen from config, so you point the
metric and ASR blocks at checkpoints for your language. The shipped presets and
example configs happen to use Russian models (it's what this was built and tested
on), but swapping in models for any other language is a config change, not a code
change.

It is built on a [datatrove](https://github.com/huggingface/datatrove)-style
block architecture (readers → metric/transcriber/labeler blocks → writers),
configured with [Hydra](https://hydra.cc), runs from a single GPU to a
multi-node cluster, and is managed with [uv](https://docs.astral.sh/uv/).

---

## Install

```bash
cd audiogear
uv sync                       # core (torch, hydra, the framework)
uv sync --extra ru-pipeline   # everything for the Russian TTS pipeline below
# or pick à-la-carte extras: --extra mos --extra asr --extra pitch --extra brouhaha ...
cp .env.example .env          # then put your HF_TOKEN in .env (gated models)
```

Requires Python 3.10–3.12 and the `espeak-ng` system package (for the speaking-rate
phonemizer): `sudo apt-get install espeak-ng`.

## Dataset structure

audiogear reads a **per-dataset `metadata.csv`** (`|`-delimited) next to an
`audio/` folder — a unified per-dataset layout under some data root:

```
<root>/
  <dataset>/                  # e.g. rootreck_fallout4, witcher, resd, css10 ...
    metadata.csv              # one row per clip, '|' separated
    audio/                    # the wavs (audio_path in the csv is relative to here's parent)
```

Point the reader at a dataset with `reader.data_folder=<root>/<dataset>` (or set
`AUDIOGEAR_DATA_DIR` once and reuse it across configs).

Minimum columns the pipeline cares about: **`id`** (unique, `"<dataset>/<rel_path>"`),
**`audio_path`** (relative to the dataset dir, e.g. `audio/NPCFCait/000A2AE7_1.wav`),
and **`text`** (may be empty — see "Annotating unlabeled audio"). Any other
columns (`dataset, domain, speaker_id, gender, emotion, duration, mos, …`) are
carried through untouched and the computed feature columns are appended. The
reader preset `configs/reader/ru.yaml` is wired for exactly this schema
(`audio_key=audio_path`, `id_key=id`, `delimiter="|"`). Bare folders of wavs with
no metadata are supported too via `reader=folder`.

## Annotating unlabeled audio (multi-ASR)

Some datasets ship audio with **no transcript** (e.g. `rootreck_fallout4` —
speaker/gender from folder names, empty `text`). `configs/annotate.yaml` fills
`text` in via **multi-ASR consensus**: it runs several Russian ASR models and
keeps the medoid hypothesis (lowest mean pairwise CER), robust to any single
model failing.

```bash
# Transcribe a whole dataset (writes the chosen transcript into `text`):
uv run python process.py --config-name annotate \
  reader.data_folder=/path/to/your/dataset

# Dry run on 10 clips:
uv run python process.py --config-name annotate reader.limit=10
```

Backends (in `configs/annotate.yaml`), all Russian-capable and open:

| Backend | Class | Install | Notes |
|---------|-------|---------|-------|
| **GigaAM** | `GigaAMBackend` | `asr` extra | pip ships v2 (`v2_rnnt`); v3 (`ai-sage/GigaAM-v3`) is manual |
| **Whisper** | `WhisperBackend` | `asr` extra | faster-whisper `large-v3` |
| **T-one** | `ToneBackend` | `uv pip install "tone @ git+https://github.com/voicekit-team/T-one.git"` | t-tech streaming Conformer-CTC |

GigaAM + Whisper are active by default; uncomment T-one in
`configs/annotate.yaml` once installed. Add another model by subclassing
`ASRBackend` and adding it to the `backends:` list. The block also emits each
model's transcript (`asr_text_<name>`) and an `asr_agreement` score; set
`min_agreement` to flag low-confidence clips (`asr_low_confidence`).

Example output (10 `rootreck_fallout4` clips, GigaAM-v2 + Whisper-large-v3):

```
[000A2AE7_1.wav] speaker=rootreck_fallout4:NPCFCait
   gigaam : охренеть прыгать оттуда очень тупо
   whisper: Охренеть! Прыгать оттуда очень тупо.
   -> CHOSEN (gigaam, agree=1.0): охренеть прыгать оттуда очень тупо
```

### Punctuation

ASR backends differ in punctuation: GigaAM-v2 emits lowercase/unpunctuated text,
while Whisper-large-v3 punctuates from the audio. The consensus saves a
**punctuated `text`** via `prefer_punctuated: true` — it still picks the medoid
for accuracy, but writes the punctuated hypothesis closest to it (so `text` keeps
Whisper's audio-derived punctuation).

A dedicated punctuation model can additionally fill a **separate column** via
`PunctuationMetric` (in `configs/annotate.yaml`):

```yaml
- _target_: audiogear.pipeline.metrics.punctuation.PunctuationMetric
  method: silero          # text-based restore (Silero TE);  or: asr (audio-based)
  column: text_punctuated
  language: ru
```

- `method: silero` — restores punctuation+casing from the transcript text
  (Silero TE; `RUPunct` is an alternative). Text-only.
- `method: asr` — re-derives punctuation from the **audio** with a punctuating
  ASR (Whisper).

> "Punctuation from text *and* audio": no single open model jointly ingests a
> reference transcript and audio — but the combination is built in. The
> **`GigaAMv3` metric** (`configs/metric/gigaam_v3.yaml`) transcribes the audio
> with **GigaAM-v3 e2e** (punctuated, normalized Russian straight from the
> acoustics) and *transfers* the heard punctuation onto the reference words:
> difflib word alignment on normalized forms, trailing `.,!?` copied per word,
> row skipped when fewer than `min_match` (60%) of words align. One GPU pass
> yields three columns: `gigaam3_text` (raw hypothesis), `gigaam3_cer`
> (agreement filter vs `text`) and `text_punctuated` (reference words + audio
> punctuation — the recommended training text). This is the pipeline that
> punctuated the VoXtream-RU corpus (~1900 h): human words stay the truth for
> phonemization, the ASR contributes the pauses and question marks it heard.

```yaml
- _target_: audiogear.pipeline.metrics.gigaam_v3.GigaAMv3
  model_name: v3_e2e_rnnt   # or v3_e2e_ctc (faster, slightly less accurate)
  punct_column: text_punctuated   # null -> transcript+CER only
  min_match: 0.6
```

## Quickstart

Processing is driven by a per-dataset config in `configs/`, selected by name:

```bash
# Dry run on 10 clips (downloads models on first use):
uv run python process.py --config-name resd reader.limit=10

# Full dataset on all GPUs:
uv run python process.py --config-name resd executor.tasks=16 executor.workers=2

# Point at your own data:
uv run python process.py --config-name resd \
  reader.data_folder=/path/to/dataset reader.glob_pattern=metadata.csv
```

Output is a CSV (one row per clip) under `outputs/`, with the original columns
plus every computed feature. Inspect the resolved config without running via
`uv run python process.py --config-name resd --cfg job`.

### Configs

```
configs/
  config.yaml            # base defaults (groups + empty metric list)
  resd.yaml              # a dataset config: --config-name resd  (declares its metrics)
  reader/   {csv,folder}.yaml
  writer/   {csv,jsonl}.yaml
  executor/ local.yaml
  metric/   one file per block (distillmos, squim, pitch, wer, ...)
```

A dataset config (e.g. `configs/resd.yaml`) selects a reader/writer/executor and
**declares which metrics to compute** as a list — each entry is resolved by
`hydra.utils.instantiate` from its `_target_`. To process a new dataset, copy
`resd.yaml` to `configs/<name>.yaml`, point the reader at your data, edit the
`metrics:` list (add/remove blocks — see `configs/metric/` for each one), and run
`uv run python process.py --config-name <name>`.

## Feature / block catalogue

Each block is a `PipelineStep`; metric blocks add columns to each clip's metadata.

| Block | Class | Columns | Backend / model | Extra |
|-------|-------|---------|-----------------|-------|
| MOS | `DistillMosMetric` | `distillmos` | DistillMOS (no-ref) | `mos` |
| Intelligibility/quality | `SquimMetrics` | `pyt_stoi`, `pyt_pesq`, `pyt_si_sdr` | torchaudio SQUIM | core |
| SNR & reverb | `SnrReverbMetrics` | `snr`, `c50` | Brouhaha (pyannote) | `brouhaha` |
| SNR (blind) | `SnrMetric` | `wada_snr` | WADA (DSP) | core |
| Bandwidth | `BandwidthMetric` | `bandwidth_hz`, `is_upsampled_est` | spectral rolloff (DSP) — catches upsampled audio the container `sample_rate` lies about | core |
| Pitch | `CrepePitchMetric` / `PitchMetric` | `pitch_mean`, `pitch_std` | torchcrepe GPU (default) / librosa pyin CPU / penn | `pitch` / core |
| Speaking rate | `SpeakingRateMetric` | `speaking_rate`, `phonemes_per_word`, `char_rate` | phonemizer (espeak) | `ru` |
| Style | `StyleMetric` | `energy_db`, `energy_dynamics`, `expressiveness` | DSP | core |
| WER/CER | `WhisperWer` | `whisper_wer`, `whisper_cer` | faster-whisper | `asr` |
| Punctuated ASR + agreement | `GigaAMv3` | `gigaam3_text`, `gigaam3_cer`, `text_punctuated`, opt-in `gigaam3_words` | GigaAM-v3 e2e (punctuation/casing derived from the audio, ru) **or** GigaAM Multilingual (`multilingual_ctc` / `multilingual_large_ctc`: ru en kk ky uz — no punctuation, set `punct_column: null`), batched, opt-in word timestamps | `asr` |
| Punctuation | `PunctuationMetric` | `text_punctuated` | Silero TE (from text) or punctuating ASR (from audio) | `asr` |
| Gender | `GenderMetric` | `gender_pred` | wav2vec2 xlsr | core |
| Emotion | `EmotionMetric` | `emotion_pred`, `emotion_score` | RU DUSHA HuBERT | core |
| Accent (EN) | `AccentMetric` | `accent` | SpeechBrain ECAPA | (speechbrain) |
| HF model (any) | `HFAudioModelMetric` | configurable | 🤗 audio model (classification/regression) | core |
| Consensus ASR | `ConsensusTranscriber` | `text`, `asr_text_*`, `asr_agreement` | GigaAM+Whisper+T-one+wav2vec2 | `asr` / `tone` |
| Speaker labeling | `SpeakerLabeler` | `speaker`, `speaker_conf`, `speaker_margin` | pyannote embed + clustering | `diarization` |
| Diarization | `DiarizationMetric` | `num_speakers`, `top_speaker_ratio` | pyannote 3.1 (gated) | `diarization` |

The per-dataset `feat_<ds>` configs enable the core subset (MOS, SQUIM, pitch,
speaking rate, WER/CER, bandwidth, style, + gender/emotion where relevant);
brouhaha, diarization, accent are config-gated.

## Two new capabilities worth calling out

- **Consensus transcription** — for clips without a transcript, run several ASR
  models (GigaAM-v2, Whisper-large-v3, a wav2vec2 model) and keep the *medoid*
  hypothesis (lowest mean pairwise CER), with an `asr_agreement` confidence.
  Robust to any single model hallucinating. Backends are pluggable — add a 4th
  by subclassing `ASRBackend`.
- **Speaker labeling with confidence thresholds** — for datasets missing speaker
  ids, embed + cluster all clips and assign an id only when it is safe
  (similarity to the cluster centroid ≥ threshold **and** a clear margin over the
  next-best cluster); otherwise the clip is left `unknown`. Precision-first, so
  you don't silently merge two speakers. Run with `executor.tasks=1` for globally
  consistent ids.

## Performance & robustness defaults

- **CPU metrics use all cores by default.** DSP metrics (`bandwidth`, `style`,
  `pitch`, `wada_snr`) fan `compute_metric` across a thread pool sized to
  `os.cpu_count()`. Override per block with `num_threads: <N>` (`-1`/`0` = all
  cores) in the metric config.
- **Batched GPU inference (length-bucketed, VRAM-bounded).** Metrics that support
  it (`gender`, `emotion`, and any `HFAudioModelMetric` — they pad with an
  attention mask, so batching is exact) sort the shard by clip length and group
  clips into batches capped by `batch_size` and `max_batch_seconds` (a VRAM proxy
  ≈ `batch_size × padded_seconds`). On OOM a batch is halved and retried (binary
  backoff); a lone long clip falls through to the per-clip recovery below. Tune
  `batch_size` / `max_batch_seconds` per block; `batch_size: 1` disables it.
- **Prefetch for bs=1 GPU metrics.** Models that can't pad-batch safely (`squim`
  has no attention mask; `distillmos` segments internally) instead decode clips
  ahead on a thread pool while inference runs single-threaded — overlapping CPU
  decode with the GPU with zero change to the values (verified bit-for-bit).
- **Models load once per worker, not once per task.** Heavy models live in a
  process-global cache, so sharding into many `tasks` no longer reloads the whole
  model stack each shard (previously minutes × tasks of overhead).
- **GPU metrics survive CUDA OOM (long clips).** A clip too long to fit in VRAM
  is retried automatically: `empty_cache` → re-decode in `chunk_seconds` (default
  20 s) windows on the GPU and aggregate → if it *still* OOMs, finish that clip on
  CPU after the GPU pass → worst case write a `NaN` sentinel. Tune per block with
  `chunk_seconds` / `cpu_overflow_threads`. Because OOM no longer kills a dataset,
  you can push `executor.workers` higher than before (a few long clips just spill
  to CPU).
- **One corrupt clip never kills a shard.** Every execution path (serial,
  parallel-CPU, per-clip GPU, batched, prefetch) routes per-clip exceptions
  through a central guard: the clip gets a sentinel value (`NaN`, or `-1` for
  WER/CER/SQUIM) plus a warning, and the run continues — treat NaN/negative
  metric values as "no value" downstream. A non-OOM error on a whole GPU batch
  retries the batch clip-by-clip so only the culprit is sentinelled. Only an
  **unbroken failure streak** (`max_consecutive_failures`, default 50) aborts
  the shard — that means a systematic problem (model load, bad config, dead
  CUDA context), not bad data. New metrics get all of this for free: implement
  `compute_metric` and do *not* wrap it in try/except.
- **Intra-shard resume (checkpoints).** Besides shard-level completion markers,
  every metric appends a per-clip JSONL checkpoint under
  `<output_folder>/checkpoints/`; a rerun after a crash resumes an unfinished
  shard from the last computed clip instead of recomputing hours of GPU work.
  On by default (`resume: false` disables, `checkpoint_dir` relocates).
  Checkpoints are keyed by clip id only — delete the `checkpoints/` folder when
  you change a metric's model or parameters.

## Execution modes (1 → N machines, 1 → N GPUs)

One sharding model covers everything: the dataset is split into `tasks` shards;
`workers` run concurrently and each worker pins one GPU. Each shard writes its
own `*_${rank}.csv`; concatenate the shards afterwards.

```bash
# 1 machine, 1 GPU
uv run python process.py --config-name resd executor.tasks=8 executor.workers=1

# 1 machine, many GPUs (one GPU per worker)
uv run python process.py --config-name resd executor.tasks=64 executor.workers=2

# Many machines (SLURM): launch the SAME command once per node; each node
# auto-claims its slice from SLURM_NODEID/SLURM_NNODES.
srun -N4 --gpus-per-node=8 \
  uv run python process.py --config-name resd executor.tasks=256 executor.workers=8

# Many machines (manual / torchrun-style): set the node env per node
AUDIOGEAR_NODE_RANK=$i AUDIOGEAR_NUM_NODES=$N \
  uv run python process.py --config-name resd executor.tasks=256 executor.workers=8
```

Runs are resumable at two levels: completed shards are skipped on rerun
(`skip_completed`), and an *unfinished* shard resumes from the last computed
clip via the per-metric checkpoints (see "Intra-shard resume" above) — rerunning
the same command after any crash is always safe. GPU detection avoids
initializing CUDA in the parent and uses `spawn`, so multi-GPU does not deadlock.

See [`docs/multi-node.md`](docs/multi-node.md) for the full distributed guide
(verified multi-GPU / multi-node / resume runs). There is **no separate Slurm
executor class** — multi-node is the *same* `LocalPipelineExecutor` launched once
per node. `build._detect_node_topology`
reads the launcher environment — `SLURM_NODEID`/`SLURM_NNODES`, torchrun
`GROUP_RANK`/`NNODES`, `NODE_RANK`/`NUM_NODES`, or explicit
`AUDIOGEAR_NODE_RANK`/`AUDIOGEAR_NUM_NODES` — and gives each node a disjoint slice
of `tasks` (it sets `local_tasks` / `local_rank_offset` for you). For cross-node
resume, put `logging_dir` (and ideally the output) on **shared storage** so every
node sees the same completion markers — local-only logs make `skip_completed` work
only within a node.

## Remote storage (S3, GCS, …)

The table I/O is [fsspec](https://filesystem-spec.readthedocs.io/)-based, so the
**input metadata CSV, the output CSVs, and the logging/checkpoint dir** can all be
remote URLs (`s3://…`, `gcs://…`, `az://…`). Install the backend (`s3fs` for S3):

```bash
uv sync --extra ru-pipeline --extra s3      # add s3fs (sync all extras together)
```

> **Audio is decoded locally.** Audio files are read with torchaudio's local
> loader, which does **not** speak `s3://`. So the *audio* must live on a
> locally-readable filesystem — local disk, or a bucket mounted via FUSE
> (`geesefs` / `goofys` / `s3fs-fuse`). The natural split is: **audio local (or
> FUSE-mounted), outputs + logs streamed to S3.** If you FUSE-mount the bucket,
> point `reader.data_folder` at the mount and everything (incl. audio) is "local".

```yaml
# audio read from local disk; results + resume markers go to S3
reader:
  data_folder: /data/mygame                 # or a FUSE mount of the bucket
writer:
  output_folder: s3://my-bucket/out/mygame
executor:
  logging_dir: s3://my-bucket/logs/mygame   # shared -> cross-node/-rerun resume
```

```bash
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_DEFAULT_REGION=...
export AWS_ENDPOINT_URL=https://storage.example.com   # for S3-compatible (MinIO/Ceph/Yandex)
```

Credentials follow the usual AWS resolution (env vars, `~/.aws/credentials`,
instance role). For a custom endpoint with the plain `s3://` string, you can also
set it via an fsspec config file (`~/.config/fsspec/conf.json`):

```json
{ "s3": { "client_kwargs": { "endpoint_url": "https://storage.example.com" } } }
```

Programmatically, `get_datafolder` also accepts a `(url, storage_options)` tuple —
e.g. `("s3://bucket/ds", {"client_kwargs": {"endpoint_url": "…"}})`. See
[`docs/storage.md`](docs/storage.md) for the full setup and a tested example.

## Extending audiogear

### Use any HuggingFace audio model — no code, just config

`HFAudioModelMetric` runs a 🤗 `AutoModelForAudioClassification` per clip and is
fully config-driven, covering both **classification** and **regression**. Add a
model straight in a dataset config's `metrics:` list — pick the model id and how
its output maps to columns:

```yaml
# classification: top-1 label string
- _target_: audiogear.pipeline.metrics.hf.HFAudioModelMetric
  model_id: alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech
  metric: gender_pred
  mode: classification
  output: label            # label | label_score | score | prob (+ `label:` for prob)
  device: ${device}

# regression: N head outputs -> N columns
- _target_: audiogear.pipeline.metrics.hf.HFAudioModelMetric
  model_id: <audio-regression-model>
  metric: [arousal, dominance, valence]
  mode: regression
  device: ${device}
```

You get batching, the process-global model cache, and the CUDA-OOM ladder for
free. `GenderMetric` / `EmotionMetric` are just thin presets over this class — copy
them for a curated default. For a fully custom block, write a `BaseMetric` instead:

### Add a new per-clip feature (metric)

1. Create `src/audiogear/pipeline/metrics/my_metric.py`:

   ```python
   from audiogear.data import AudioSegment
   from audiogear.pipeline.metrics.base import BaseMetric

   class MyMetric(BaseMetric):
       name = "✨ MyMetric"
       _requires_dependencies = ("some_pkg",)        # checked at construction

       def __init__(self, device="cuda", file_writer=None, file_reader=None):
           # one column -> a str; several -> a tuple of column names
           super().__init__(metric="my_feature", file_writer=file_writer, file_reader=file_reader)
           self.device = device
           self._model = None                          # lazy: load on first use

       @property
       def model(self):
           if self._model is None:
               import some_pkg
               self._model = some_pkg.load().to(self.device)
           return self._model

       def compute_metric(self, segment: AudioSegment):
           return float(self.model(segment.audio_file))   # -> the column value
   ```

   Inheritance: `PipelineStep` (dependency check, `run`) → `BaseMetric`
   (`compute_metric`, checkpoint/resume) → `MyMetric`. Load models **lazily** and
   read `self.device` — never set `CUDA_VISIBLE_DEVICES` yourself (the executor
   pins GPUs). Use `audiogear.audio.load_audio(path, target_sr=...)` for I/O.

2. Add `configs/metric/my_metric.yaml` (for reuse/reference):

   ```yaml
   _target_: audiogear.pipeline.metrics.my_metric.MyMetric
   device: ${device}
   ```

3. Enable it by adding it to the `metrics:` list of a dataset config (e.g.
   `configs/resd.yaml`) with its `_target_`.

### Add an ASR backend to the consensus

The consensus transcriber (see below) ensembles any number of `ASRBackend`s. A
backend wraps one model behind `transcribe(path) -> str` with **lazy, cached**
loading (so it pickles cheaply to a worker and loads once per process):

```python
from audiogear.pipeline.transcribers.base import ASRBackend

class MyASRBackend(ASRBackend):
    backend_name = "myasr"                 # -> per-clip column `asr_text_myasr`

    def __init__(self, model_id="org/model", name=None, device="cuda"):
        super().__init__(name=name, device=device)
        self.model_id = model_id

    def _cache_key(self):                  # so distinct checkpoints cache separately
        return (type(self).__name__, self.model_id, self.device)

    def _load(self):                       # built once per worker; self.model caches it
        import my_lib
        return my_lib.load(self.model_id).to(self.device)

    def transcribe(self, audio_file: str) -> str:
        return self.model.transcribe(audio_file)
```

Then list it under `backends:` in `configs/annotate.yaml` (or any config that uses
`ConsensusTranscriber`). Built-ins: `GigaAMBackend`, `WhisperBackend`,
`Wav2Vec2Backend`, `ToneBackend`. A backend that fails to *load* (e.g. an optional
one that isn't installed) is disabled after one warning instead of crashing the
run; a per-clip failure just drops that hypothesis.

### How the consensus transcriber works

`ConsensusTranscriber` runs every backend on a clip and picks the **medoid** — the
hypothesis with the lowest mean pairwise CER to the others (a correct transcript
is close to the other correct ones; a hallucination sits far from the pack). It
writes:

- `asr_text_<name>` — each backend's raw transcript,
- `text` — the chosen transcript (when `overwrite_text: true`),
- `asr_chosen_backend`, `asr_agreement` (1 = identical), and `asr_low_confidence`
  when `min_agreement` is set.

`only_missing: true` transcribes only clips with empty `text`. `prefer_punctuated:
true` keeps the medoid for *scoring* but saves the punctuated hypothesis closest
to it into `text` (so you keep audio-derived punctuation from e.g. Whisper, even
when the medoid is an unpunctuated model like GigaAM v2). See `configs/annotate.yaml`
for a ready 2–3 backend setup.

### Add a reader, writer, or dataset
- Reader/writer: subclass `BaseDiskReader` / `DiskWriter` and add a `configs/reader`
  or `configs/writer` yaml. For a new dataset, usually just point the CSV reader at
  its metadata (`reader.data_folder`, `reader.audio_key`, `reader.delimiter`).

## Processing many datasets

For a one-off dataset, `process.py --config-name <name>` is enough. To sweep a
whole **collection** of datasets — generate a per-dataset config for each, run
them smallest-first with resume, merge the per-shard CSVs (with id dedup), then
filter to a clean subset — see [`examples/`](examples/): config generation
(`gen_configs.py`), a batch runner (`run_batch.py`), a quality filter
(`filter_clean.py`), and a QA report (`qa_report.py`), all driven by
`AUDIOGEAR_DATA_DIR` (no hardcoded paths). They are templates to copy and adapt,
not a fixed CLI.

### QA report & threshold calibration

```bash
export AUDIOGEAR_DATA_DIR=/path/to/data_root
uv run python examples/filter_clean.py resd   # extended_metadata.csv -> clean_metadata.csv + drop reasons
uv run python examples/qa_report.py resd      # -> <data_root>/resd/qa_report.html
```

`qa_report.py` renders one **self-contained HTML per dataset**: summary cards,
the drop-reason breakdown, an SVG histogram per filtered metric with the
`filter_clean` threshold drawn in, and **playable audio embedded for the clips
just below / just above each threshold**. That is the threshold-calibration
loop: if the rejected clips near a threshold sound fine, the threshold is too
strict; if the accepted ones sound bad, it is too loose — adjust `TH` in
`examples/filter_clean.py` (the single source of truth for both the filter and
the report) and regenerate.

## Using audiogear from another project (agent skill)

[`SKILL.md`](SKILL.md) is a ready-made [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills)
for Claude Code: copy it into a consuming project as
`.claude/skills/audiogear/SKILL.md` and the agent knows how to install
audiogear, prepare `metadata.csv`, write a per-dataset config, run/resume the
pipeline, merge shards, filter, and build QA reports — including the pitfalls
(lane rules, sentinel semantics, checkpoint staleness).

## Testing

```bash
uv sync --extra dev        # pytest
uv run pytest              # or: .venv/bin/python -m pytest
```

The suite is **CPU-only and downloads no model weights** — it exercises the
pipeline plumbing where the real bugs live: CSV writer/reader round-trips (column
alignment), result↔segment pairing across every scheduling path (batched /
prefetch / plain-GPU / parallel-CPU), the per-clip failure guard on all of those
paths (poisoned clips → sentinels, streak abort), checkpoint resume (partial
runs, torn lines, type round-trips), shard-merge dedup, the runtime helpers
(length bucketing, windowing, OOM detection), and text normalization. Fast
enough to run on every change. See [`docs/testing.md`](docs/testing.md) for the
suite plus a verified end-to-end run on a small subset (2× RTX 4090).

## Project layout

```
audiogear/
  process.py                 # entrypoint: python process.py --config-name <dataset>
  SKILL.md                   # agent skill: drive audiogear from another project
  configs/                   # Hydra configs (config.yaml, resd.yaml, reader/ writer/ executor/ metric/)
  src/audiogear/
    audio.py                 # shared load/resample
    data.py                  # AudioSegment / AudioPipeline
    build.py                 # Hydra builder (instantiate -> pipeline -> executor, checkpoint wiring)
    pipeline/
      readers/ writers/ segmenters/
      metrics/               # all metric blocks
      transcribers/          # consensus ASR + backends
      parallel.py            # ParallelLanes: concurrent CPU∥GPU metric lanes
      checkpoint.py          # per-metric JSONL checkpoints (intra-shard resume)
    executer/                # Local executor (GPU pinning, sharding, resume)
    utils/                   # runtime.py (threads, CUDA-OOM ladder, model cache, batching), progress.py
  tests/                     # pytest suite (CPU-only, no model downloads)
  docs/                      # multi-node.md, storage.md (S3), testing.md
  examples/                  # multi-dataset templates (config gen, batch run, filtering, QA report)
  eval/                      # sanity_distillmos.py
  models/                    # downloaded model weights (git-ignored)
```

## Notes on dependency pins
`gigaam` installs **from git, pinned to a commit** (PyPI is stuck at 0.1.0,
v2-era: no v3/e2e or multilingual models and a hard `torchaudio<=2.5.1` pin).
gigaam 0.2 dropped that pin, but torch stays capped `<2.6` — it is what this
stack is tested against; transformers is capped `<5` and huggingface-hub `<1.0`
to keep pyannote 3.x working; a uv `override-dependencies` forces a loadable
`onnxruntime`. transformers-based ASR models must therefore ship `safetensors`.
See `pyproject.toml` for the rationale.

## Write-ups

Longer-form articles on the design and the performance work (length-bucketed GPU
batching, the CUDA-OOM recovery ladder, prefetch, parallel CPU∥GPU lanes, the
model cache) live in [`articles/`](articles/) — English ([`medium_en.md`](articles/medium_en.md))
and Russian ([`habr_ru.md`](articles/habr_ru.md)).

## License

[MIT](LICENSE).
