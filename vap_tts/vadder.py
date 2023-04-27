import torch


class VadSilero:
    def __init__(
        self,
        onnx=True,
        min_speech_duration=0.1,
        min_silence_duration=0.05,
        speech_pad=0,
        sample_rate=16_000,
    ) -> None:
        self.min_speech_duration_ms = int(min_speech_duration * 1000)
        self.min_silence_duration_ms = int(min_silence_duration * 1000)
        self.speech_pad_ms = int(speech_pad * 1000)
        self.window_size_samples = 512

        # Model
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=onnx,
        )
        self.get_speech_timestamps = utils[0]
        self.sample_rate = sample_rate

    def vad_list(self, waveform):
        assert (
            waveform.ndim == 1
        ), f"Silero.vad_list: Wrong waveform shape expected (1, n_samples) got {waveform.shape}"
        ch_list = self.get_speech_timestamps(
            waveform,
            self.model,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            window_size_samples=self.window_size_samples,
            return_seconds=True,
            sampling_rate=self.sample_rate,
        )
        vad_list = []
        for x in ch_list:
            s = x["start"]
            e = x["end"]
            vad_list.append([s, e])
        return vad_list

    def get_vad_list(self, waveform):
        assert (
            waveform.ndim == 2
        ), f"Silero.get_vad_list: Wrong waveform shape expected (2, n_samples) got {waveform.shape}"
        vad_list = [[], []]
        for ch in range(2):
            vad_list[ch] = self.vad_list(waveform[ch])
        return vad_list
