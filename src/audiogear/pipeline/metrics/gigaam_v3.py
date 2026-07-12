"""GigaAM-v3 e2e transcription: punctuated Russian text + CER vs reference.

Why a dedicated metric: GigaAM-v3 ``v3_e2e_*`` heads emit NORMALIZED text with
punctuation and casing derived from the ACOUSTICS (pauses, question intonation)
— the closest open model to "punctuation by audio" (its training punctuation
was produced by GigaChat Max Audio from the audio track). One pass gives both:

- ``gigaam3_text`` — punctuated hypothesis, a source to transfer punctuation
  onto a reference transcript (word-align, copy trailing marks);
- ``gigaam3_cer``  — agreement filter vs ``segment.text`` (same normalization
  as ``whisper_cer``; ``-1`` when the segment has no reference text).

Batched: encoder forward + RNNT/CTC decode both take real batches.
"""

from __future__ import annotations

from audiogear.audio import load_audio
from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.pipeline.metrics.wer import normalize_text
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter
from audiogear.utils.runtime import cached_model, normalize_device

SAMPLE_RATE = 16000
MAX_SECONDS = 24.9  # gigaam short-form limit is 25 s; longer clips are truncated


class GigaAMv3(BaseMetric):
    """Transcribe with GigaAM-v3 (default ``v3_e2e_rnnt``) -> text + CER."""

    name = "🗣 GigaAM-v3"
    gpu = True
    supports_batch = True
    _requires_dependencies = ("gigaam", "jiwer")

    def __init__(
        self,
        model_name: str = "v3_e2e_rnnt",
        device: str = "cuda",
        text_column: str = "gigaam3_text",
        cer_column: str = "gigaam3_cer",
        batch_size: int = 16,
        max_batch_seconds: float = 320.0,
        chunk_seconds: float = 20.0,
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        super().__init__(
            metric=(text_column, cer_column),
            file_writer=file_writer,
            file_reader=file_reader,
            chunk_seconds=chunk_seconds,
            batch_size=batch_size,
            max_batch_seconds=max_batch_seconds,
        )
        self.model_name = model_name
        self.device = device

    def _failed_value(self):
        # The text column is a string — an empty hypothesis (not NaN) keeps the
        # CSV typed; CER -1 matches the "could not score" convention.
        return "", -1.0

    def _model_on(self, device: str):
        def build():
            import gigaam

            return gigaam.load_model(self.model_name, device=device)

        return cached_model(("GigaAMv3", self.model_name, device), build)

    def _load_1d(self, segment: AudioSegment):
        audio, _ = load_audio(segment.audio_file, target_sr=SAMPLE_RATE, mono=True)
        return audio.squeeze(0)[: int(MAX_SECONDS * SAMPLE_RATE)]

    def _transcribe(self, wavs: list, device: str) -> list[str]:
        import torch

        model = self._model_on(device)
        lens = torch.tensor([w.shape[-1] for w in wavs])
        batch = torch.zeros(len(wavs), int(lens.max()))
        for i, w in enumerate(wavs):
            batch[i, : w.shape[-1]] = w
        batch = batch.to(device)
        with torch.inference_mode():
            enc, enc_len = model.forward(batch, lens.to(device))
            return [t for t, _, _ in model.decoding.decode(model.head, enc, enc_len)]

    def _score(self, segment: AudioSegment, hypothesis: str):
        import jiwer

        reference = normalize_text(segment.text or "")
        if not reference:
            return hypothesis, -1.0
        return hypothesis, float(jiwer.cer(reference, normalize_text(hypothesis)))

    def compute_batch(self, segments: list[AudioSegment]):
        device = normalize_device(self.device)
        texts = self._transcribe([self._load_1d(s) for s in segments], device)
        return [self._score(s, t) for s, t in zip(segments, texts)]

    def compute_metric(self, segment: AudioSegment):
        return self.compute_batch([segment])[0]

    def compute_metric_cpu(self, segment: AudioSegment):
        texts = self._transcribe([self._load_1d(segment)], "cpu")
        return self._score(segment, texts[0])
