"""GigaAM transcription: punctuated text + CER + punctuation transfer.

Why a dedicated metric: GigaAM-v3 ``v3_e2e_*`` heads emit NORMALIZED text with
punctuation and casing derived from the ACOUSTICS (pauses, question intonation)
— the closest open model to "punctuation by audio" (its training punctuation
was produced by GigaChat Max Audio from the audio track). One pass gives:

- ``gigaam3_text`` — punctuated hypothesis, a source to transfer punctuation
  onto a reference transcript (word-align, copy trailing marks);
- ``gigaam3_cer``  — agreement filter vs ``segment.text`` (same normalization
  as ``whisper_cer``; ``-1`` when the segment has no reference text);
- ``text_punctuated`` — ``segment.text`` WORDS with the hypothesis punctuation
  transferred onto them (``transfer_punctuation``): human words stay the truth
  for phonemization, the ASR contributes the pauses/question marks it heard.
  Empty when there is no reference or fewer than ``min_match`` words aligned.
- ``gigaam3_words`` (opt-in via ``words_column``) — word-level timestamps of
  the hypothesis as JSON ``[{"text", "start", "end"}, ...]`` (seconds), for
  audio↔word alignment tooling downstream.

The same class runs the whole GigaAM family (``model_name``): the new
``multilingual_ctc`` / ``multilingual_large_ctc`` (220M / 600M; ru, en, kk, ky,
uz) transcribe more languages but are charwise CTC — lowercase, NO punctuation —
so pair them with ``punct_column: null`` and use them for coverage/agreement,
not as a punctuation source (that stays ``v3_e2e_*``, Russian only). See
``configs/metric/gigaam_multilingual.yaml``.

Batched: encoder forward + RNNT/CTC decode both take real batches. Requires
gigaam >= 0.2 (git): it adds the multilingual models and word timestamps, and
fixes conv-padding leakage in batched encoding, so batched results match
single-clip runs.
"""

from __future__ import annotations

import json

from audiogear.audio import load_audio
from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.pipeline.metrics.punctuation import transfer_punctuation
from audiogear.pipeline.metrics.wer import normalize_text
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter
from audiogear.utils.runtime import cached_model, normalize_device

SAMPLE_RATE = 16000
MAX_SECONDS = 24.9  # gigaam short-form limit is 25 s; longer clips are truncated


class GigaAMv3(BaseMetric):
    """Transcribe with GigaAM (default ``v3_e2e_rnnt``) -> text + CER (+ punct/words)."""

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
        punct_column: str | None = "text_punctuated",
        min_match: float = 0.6,
        words_column: str | None = None,
        batch_size: int = 16,
        max_batch_seconds: float = 320.0,
        chunk_seconds: float = 20.0,
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        """punct_column: column for ``segment.text`` with the hypothesis
        punctuation transferred onto it; ``null`` disables the third column.
        min_match: minimum fraction of reference words that must align with the
        hypothesis for the transfer to be trusted (below it -> empty string).
        words_column: column for word-level timestamps of the hypothesis (JSON);
        ``null`` (default) disables."""
        metric = (text_column, cer_column)
        if punct_column:
            metric += (punct_column,)
        if words_column:
            metric += (words_column,)
        super().__init__(
            metric=metric,
            file_writer=file_writer,
            file_reader=file_reader,
            chunk_seconds=chunk_seconds,
            batch_size=batch_size,
            max_batch_seconds=max_batch_seconds,
        )
        self.model_name = model_name
        self.device = device
        self.punct_column = punct_column
        self.min_match = min_match
        self.words_column = words_column

    def _failed_value(self):
        # The string columns get empty strings (not NaN) so the CSV stays typed;
        # CER -1 matches the "could not score" convention.
        failed = ("", -1.0)
        if self.punct_column:
            failed += ("",)
        if self.words_column:
            failed += ("[]",)
        return failed

    def _model_on(self, device: str):
        def build():
            import gigaam

            return gigaam.load_model(self.model_name, device=device)

        return cached_model(("GigaAMv3", self.model_name, device), build)

    def _load_1d(self, segment: AudioSegment):
        audio, _ = load_audio(segment.audio_file, target_sr=SAMPLE_RATE, mono=True)
        return audio.squeeze(0)[: int(MAX_SECONDS * SAMPLE_RATE)]

    def _transcribe(self, wavs: list, device: str) -> list[tuple[str, list | None]]:
        """Batched forward + decode -> [(text, words | None), ...]."""
        import torch

        model = self._model_on(device)
        lens = torch.tensor([w.shape[-1] for w in wavs])
        batch = torch.zeros(len(wavs), int(lens.max()))
        for i, w in enumerate(wavs):
            batch[i, : w.shape[-1]] = w
        batch = batch.to(device)
        with torch.inference_mode():
            enc, enc_len = model.forward(batch, lens.to(device))
            # _decode wraps decoding.decode and, when asked, converts the
            # decoder's token frames into word timestamps (seconds).
            return model._decode(enc, enc_len, wav_lens=lens, word_timestamps=bool(self.words_column))

    def _score(self, segment: AudioSegment, hypothesis: str, words: list | None = None):
        import jiwer

        reference = normalize_text(segment.text or "")
        cer = -1.0 if not reference else float(jiwer.cer(reference, normalize_text(hypothesis)))
        result = (hypothesis, cer)
        if self.punct_column:
            transferred, _ = transfer_punctuation(segment.text or "", hypothesis, self.min_match)
            result += (transferred or "",)
        if self.words_column:
            result += (json.dumps(
                [{"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)} for w in words or []],
                ensure_ascii=False,
            ),)
        return result

    def compute_batch(self, segments: list[AudioSegment]):
        device = normalize_device(self.device)
        decoded = self._transcribe([self._load_1d(s) for s in segments], device)
        return [self._score(s, text, words) for s, (text, words) in zip(segments, decoded)]

    def compute_metric(self, segment: AudioSegment):
        return self.compute_batch([segment])[0]

    def compute_metric_cpu(self, segment: AudioSegment):
        text, words = self._transcribe([self._load_1d(segment)], "cpu")[0]
        return self._score(segment, text, words)
