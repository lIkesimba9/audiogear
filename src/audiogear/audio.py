"""Shared audio loading / resampling helpers.

Many metric blocks need a waveform at a specific sample rate (SQUIM and
brouhaha want 16 kHz, DistillMOS wants 16 kHz, Whisper wants 16 kHz, penn is
sample-rate aware). Historically each block re-implemented load+resample, which
was both duplicated and slow (librosa's default resampler). This module
centralises it on torchaudio with cached resamplers (the "faster resampling
with torchaudio" approach ported from the dev branch).

Everything here imports torch/torchaudio lazily so that importing
``audiogear`` stays light.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


@lru_cache(maxsize=32)
def _get_resampler(orig_sr: int, target_sr: int):
    import torchaudio

    return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)


def load_audio(
    path: str,
    target_sr: int | None = None,
    mono: bool = True,
) -> tuple["torch.Tensor", int]:
    """Load an audio file as a float32 torch tensor.

    Args:
        path: path to the audio file.
        target_sr: if given, resample to this rate (Hz). If ``None``, keep
            the file's native rate.
        mono: if True, average channels down to a single channel. The returned
            tensor is always shape ``(channels, samples)`` (channels == 1 when
            ``mono``), matching torchaudio's convention.

    Returns:
        ``(waveform, sample_rate)`` where ``sample_rate`` is the rate *after*
        any resampling.
    """
    import torch
    import torchaudio

    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        # torchaudio's ffmpeg backend chokes on some valid files (e.g. MP3s whose
        # default_audio_stream is None -> get_src_stream_info(None)); soundfile
        # (libsndfile) reads them fine. Fall back so one odd file doesn't kill a shard.
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
        waveform = torch.from_numpy(data.T).contiguous()           # (channels, frames)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        waveform = _get_resampler(sr, target_sr)(waveform)
        sr = target_sr
    # Ensure float32 in [-1, 1]
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)
    return waveform, sr


def audio_duration(path: str) -> float:
    """Clip duration in seconds, read from the header only (no full decode).

    Used by the length-bucketed GPU batcher to size batches. Falls back to a
    full decode if the backend cannot report frame count from metadata, and to
    ``0.0`` if the file is unreadable (it then sorts first / batches harmlessly).
    """
    import torchaudio

    try:
        info = torchaudio.info(path)
        if info.num_frames > 0 and info.sample_rate > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass
    try:
        wav, sr = torchaudio.load(path)
        return wav.shape[-1] / sr if sr else 0.0
    except Exception:
        pass
    try:  # soundfile reads files torchaudio's ffmpeg backend rejects
        import soundfile as sf

        info = sf.info(path)
        return info.frames / info.samplerate if info.samplerate else 0.0
    except Exception:
        return 0.0


def resample(waveform: "torch.Tensor", orig_sr: int, target_sr: int) -> "torch.Tensor":
    """Resample an already-loaded waveform tensor, using a cached resampler."""
    if orig_sr == target_sr:
        return waveform
    return _get_resampler(orig_sr, target_sr)(waveform)
