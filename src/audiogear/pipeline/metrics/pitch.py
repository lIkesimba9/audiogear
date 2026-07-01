from audiogear.audio import load_audio
from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric, PrefetchGPUMetric


class PitchMetric(BaseMetric):
    """Fundamental-frequency (pitch) statistics: mean and std over voiced frames.

    Mirrors DataSpeech's pitch + "speech monotony" signal. This class holds the
    non-prefetch backends; pick one with ``backend=``:

    - ``pyin`` (default): librosa's probabilistic YIN. CPU-only, zero extra
      dependencies (librosa is a core dep), fanned across threads
      (``parallel_cpu``). The safe no-GPU option.
    - ``penn``: the neural `penn` estimator (GPU, per-clip). Needs the ``pitch``
      extra; its compiled ``torbi`` dependency can be fragile against specific
      torch builds (it may fail to load).

    For **GPU pitch, prefer** :class:`CrepePitchMetric` (neural CREPE) — it runs on
    the GPU through the decode-prefetch + CUDA-OOM-ladder machinery instead of the
    naive per-clip loop, installs cleanly (no torbi), and is the default pitch
    block in the shipped configs.

    Emits ``pitch_mean`` and ``pitch_std`` (Hz), over voiced frames only (0.0 if
    the clip has no voiced content).
    """

    name = "🎵 Pitch"

    parallel_cpu = True

    def __init__(
        self,
        backend: str = "pyin",
        fmin: float = 65.0,
        fmax: float = 1000.0,
        hopsize: float = 0.01,
        gpu: int | None = None,
        model_path: str = None,
        center: str = "half-hop",
        batch_size: int = 1,
        num_threads: int = -1,
        file_writer=None,
        file_reader=None,
    ):
        super().__init__(metric=("pitch_mean", "pitch_std"), file_writer=file_writer, file_reader=file_reader, num_threads=num_threads)
        self.backend = backend
        self.fmin = fmin
        self.fmax = fmax
        self.hopsize = hopsize
        self.gpu = gpu
        self.model_path = model_path
        self.center = center
        self.batch_size = batch_size
        # penn runs a single CUDA stream -> NOT thread-fanned (thread fan-out is
        # only for the GIL-bound CPU pyin path).
        if backend == "penn":
            self.parallel_cpu = False

    def _compute_pyin(self, segment: AudioSegment):
        import librosa
        import numpy as np

        y, sr = librosa.load(segment.audio_file, sr=None, mono=True)
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=sr, fmin=self.fmin, fmax=self.fmax, frame_length=2048
        )
        voiced = f0[voiced_flag & ~np.isnan(f0)]
        if voiced.size == 0:
            return 0.0, 0.0
        return float(np.mean(voiced)), float(np.std(voiced))

    def _compute_penn(self, segment: AudioSegment):
        import penn
        import torch

        audio, sr = load_audio(segment.audio_file, mono=True)
        pitch, periodicity = penn.from_audio(
            audio,
            sr,
            hopsize=self.hopsize,
            fmin=self.fmin,
            fmax=self.fmax,
            checkpoint=self.model_path,
            center=self.center,
            gpu=self.gpu,
            batch_size=self.batch_size,
        )
        return torch.mean(pitch).item(), torch.std(pitch).item()

    def compute_metric(self, segment: AudioSegment):
        if self.backend == "penn":
            return self._compute_penn(segment)
        return self._compute_pyin(segment)


class CrepePitchMetric(PrefetchGPUMetric):
    """GPU pitch via torchcrepe (neural CREPE), run through the prefetch pipeline.

    Same ``pitch_mean`` / ``pitch_std`` as :class:`PitchMetric`, but this is a
    proper GPU metric: audio is decoded ahead on a thread pool while CREPE
    inference runs on a single CUDA stream (``prefetch``), so the GPU stays busy
    instead of waiting on per-clip decode, and long clips fall through the
    inherited CUDA-OOM ladder (windowed GPU -> CPU -> sentinel). torchcrepe has no
    compiled ``torbi`` (penn's fragile dep), so it installs cleanly against any
    torch build (``pip install torchcrepe`` / the ``pitch`` extra).

    ``crepe_model`` picks the ``"tiny"`` (fast, default) or ``"full"`` (more
    accurate) weights; ``periodicity`` is the voiced-frame confidence threshold.
    Audio is resampled to 16 kHz (CREPE's rate) on decode.
    """

    name = "🎵 Pitch (CREPE)"
    sample_rate = 16000
    _requires_dependencies = ("torchcrepe", "torch")

    def __init__(
        self,
        device: str = "cuda",
        crepe_model: str = "tiny",
        fmin: float = 65.0,
        fmax: float = 1000.0,
        hopsize: float = 0.01,
        periodicity: float = 0.5,
        batch_size: int = 512,
        chunk_seconds: float = 20.0,
        file_writer=None,
        file_reader=None,
    ):
        super().__init__(
            metric=("pitch_mean", "pitch_std"),
            device=device,
            chunk_seconds=chunk_seconds,
            file_writer=file_writer,
            file_reader=file_reader,
        )
        self.crepe_model = crepe_model
        self.fmin = fmin
        self.fmax = fmax
        self.hopsize = hopsize
        self.periodicity = periodicity
        self.batch_size = batch_size

    def _model_on(self, device: str):
        # torchcrepe caches its network globally per (capacity, device); there is
        # no model object to build/hold on the instance.
        return None

    def _run(self, audio, device: str):
        import torchcrepe

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (samples,) -> (1, samples)
        hop = max(1, int(round(self.sample_rate * self.hopsize)))
        f0, pd = torchcrepe.predict(
            audio,
            self.sample_rate,
            hop_length=hop,
            fmin=self.fmin,
            fmax=self.fmax,
            model=self.crepe_model,
            device=device,
            batch_size=self.batch_size,
            return_periodicity=True,
            pad=True,
        )
        voiced = f0[pd > self.periodicity]
        if voiced.numel() == 0:
            return 0.0, 0.0
        return float(voiced.mean().item()), float(voiced.std().item())
