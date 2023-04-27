"""
Postprocessing
--------------

Transform the last word (first and last sentence) to be either 
more turn-yielding or more turn-taking.

We extract data that only contains a single syllable in both the last word of 
the first sentence and the last word of the last sentence.


PRAAT
- https://www.fon.hum.uva.nl/praat/manual/Intro_8_3__Manipulation_of_intensity.html
- https://www.fon.hum.uva.nl/praat/manual/Intro_8_1__Manipulation_of_pitch.html
"""

import torch
import torchaudio.functional as AF
import numpy as np
import parselmouth
from os.path import join
from parselmouth.praat import call
from parselmouth import Sound

from vap_tts.utils import load_audio, load_text_grid

SAMPLE_RATE: int = 16_000
HOP_TIME: float = 0.02
F0_MIN: int = 60
F0_MAX: int = 400


# TODO: Extract Pitch, Intensity, duration/speech-rate
# Apply turn-hold  prosody -> flat pitch, higher intensity, longer duration?
# Apply turn-shift prosody -> rising pitch, lower intensity, shorter duration?


def torch_to_praat_sound(x, sample_rate: int = SAMPLE_RATE) -> Sound:
    if isinstance(x, Sound):
        return x

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if x.dtype != np.float64:
        x = x.astype("float64")
    return parselmouth.Sound(x, sampling_frequency=sample_rate)


def praat_to_torch(sound):
    y = sound.as_array().astype("float32")
    return torch.from_numpy(y)


def get_single_pitch(
    sound, sample_rate=SAMPLE_RATE, f0_min=F0_MIN, f0_max=F0_MAX, hop_time=HOP_TIME
):
    if isinstance(sound, torch.Tensor):
        assert sound.ndim == 1, f"Expects (n_samples) got {tuple(sound.shape)}"
    sound = torch_to_praat_sound(sound, sample_rate)
    pitch = sound.to_pitch(time_step=hop_time, pitch_floor=f0_min, pitch_ceiling=f0_max)
    return torch.from_numpy(pitch.selected_array["frequency"]).float()


def get_single_intensity(
    sound, sample_rate=SAMPLE_RATE, f0_min=F0_MIN, hop_time=HOP_TIME, subtract_mean=True
):
    if isinstance(sound, torch.Tensor):
        assert sound.ndim == 1, f"Expects (n_samples) got {tuple(sound.shape)}"

    sound = torch_to_praat_sound(sound, sample_rate)
    intensity = sound.to_intensity(
        minimum_pitch=f0_min, time_step=hop_time, subtract_mean=subtract_mean
    ).as_array()[0]
    return torch.from_numpy(intensity).float()


def rms(
    x: torch.Tensor,
    hop_time: float = 0.02,
    frame_time: float = 0.05,
    sample_rate: int = SAMPLE_RATE,
    window: str = "hann",
):
    """
    Root-mean-square (RMS) energy of a signal.

    Based on:
        https://librosa.org/doc/latest/generated/librosa.feature.rms.html#librosa.feature.rms
    """
    assert x.ndim == 1, f"Expects (n_samples) got {tuple(x.shape)}"

    assert window in ["hann", "hamming"]

    frame_samples = int(frame_time * sample_rate)
    hop_samples = int(hop_time * sample_rate)
    frames = audio.unfold(dimension=0, size=frame_samples, step=hop_samples)

    w = torch.hann_window(frame_samples)
    if window == "hamming":
        w = torch.hamming_window(frame_samples)
    frames = frames * w
    return frames.pow(2).mean(dim=1).sqrt()


def find_hold_times(d, tg):
    """
    Find the word timings of the last word in 'first' or 'last' sentence

    This is quite an ugly function, but it works. We match the corresponding bi-gram
    in the sentences with the TextGrid words.
    """

    target_word = d["sentences"][0].split(" ")[-1].lower()
    post_silence_word = d["sentences"][1].split(" ")[0].lower()

    wstart, wend = -1, -1
    send = -1
    # Find where the last-word is followed by silence
    for widx in range(len(tg["words"]["words"]) - 1):
        next_idx = widx + 1
        cur_word = tg["words"]["words"][widx]
        next_word = tg["words"]["words"][next_idx]
        if cur_word == target_word and next_word == "":
            wstart = tg["words"]["starts"][widx]
            wend = tg["words"]["ends"][widx]

        if cur_word == "" and next_word == post_silence_word:
            send = tg["words"]["starts"][next_idx]

    if wend > send:
        wstart = -1
        wend = -1
        send = -1
    return {
        "word": {"start": wstart, "end": wend},
        "silence": {"start": wend, "end": send},
    }


def flatten_pitch(
    sound: Sound,
    start: float,
    end: float,
    pre_pitch_time: float = 0.5,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
    f0_max: int = F0_MAX,
    sample_rate: int = SAMPLE_RATE,
) -> Sound:
    """
    Flatten the pitch of the last word in the first sentence.

    Use the `pre_pitch_time` to get the average pitch value prior to the `start` time.
    The target for the flattening.
    """

    # Get the average pitch value of the last 0.5 seconds prior to last word
    pitch = get_single_pitch(sound, sample_rate, f0_min, f0_max, hop_time)
    pre_start = start - pre_pitch_time
    pre_start = pre_start if pre_start > 0 else 0
    pre_frame = int(pre_start / hop_time)
    end_frame = int(start / hop_time)
    p = pitch[pre_frame:end_frame]
    target_f0 = p[p > 0].mean().item()

    # https://www.fon.hum.uva.nl/praat/manual/Intro_8_1__Manipulation_of_pitch.html
    manipulation = call(sound, "To Manipulation", hop_time, f0_min, f0_max)

    # pitch
    pitch_tier = call(manipulation, "Create PitchTier", "flat", start, end)
    call(pitch_tier, "Add point", start, target_f0)
    call(pitch_tier, "Add point", end, target_f0)

    # Select the original and the replacement tier -> replace pitch
    call([pitch_tier, manipulation], "Replace pitch tier")

    # Extract the new sound
    return call(manipulation, "Get resynthesis (overlap-add)")


def lengthening(
    sound: Sound,
    start: float,
    end: float,
    scale=1.1,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
    f0_max: int = F0_MAX,
) -> Sound:
    """
    https://www.fon.hum.uva.nl/praat/manual/Intro_8_2__Manipulation_of_duration.html
    https://www.fon.hum.uva.nl/praat/manual/DurationTier.html

    One of the types of objects in Praat. A DurationTier object contains a
    number of (time, duration) points, where duration is to be interpreted
    as a relative duration (e.g. the duration of a manipulated sound as
    compared to the duration of the original). For instance, if
    your DurationTier contains two points, one with a duration value of 1.5
    at a time of 0.5 seconds and one with a duration value of 0.6 at a time
    of 1.1 seconds, this is to be interpreted as a relative duration of 1.5
    (i.e. a slowing down) for all original times before 0.5 seconds, a
    relative duration of 0.6 (i.e. a speeding up) for all original times
    after 1.1 seconds, and a linear interpolation between 0.5 and 1.1
    seconds (e.g. a relative duration of 1.2 at 0.7 seconds, and of 0.9 at 0.9 seconds).
    """

    manipulation = call(sound, "To Manipulation", hop_time, f0_min, f0_max)
    dur_tier = call(
        manipulation,
        "Create DurationTier",
        "shorten",
        0,
        sound.end_time,
    )

    eps = 0.01

    # before this point duration should be the same
    call(dur_tier, "Add point", start - eps, 1.0)

    # Change duration here
    call(dur_tier, "Add point", start, scale)
    call(dur_tier, "Add point", end, scale)

    # After this point duration should be the same
    call(dur_tier, "Add point", end + eps, 1.0)

    call([manipulation, dur_tier], "Replace duration tier")
    sound_dur = call(manipulation, "Get resynthesis (overlap-add)")
    return sound_dur


def raise_intensity(
    sound: Sound, start: float, end: float, scale=2.0, sample_rate: int = SAMPLE_RATE
) -> torch.Tensor:
    """
    Could not get praat intensity to work... Relative scale in decibels/pascal was the problem
    Scaling the amplitudes of the waveform directly instead... very simple...

    This should be the last transform so we can return a torch.Tensor
    """
    # https://www.fon.hum.uva.nl/praat/manual/Intro_8_3__Manipulation_of_intensity.html
    sound = praat_to_torch(sound)[0]

    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    n_samples = end_sample - start_sample
    n1 = int(n_samples / 2)
    n2 = n_samples - n1
    scale1 = torch.linspace(1.1, scale, n1)
    scale2 = torch.tensor([scale] * n2)
    scale = torch.cat((scale1, scale2))
    sound[start_sample:end_sample] *= scale
    return sound


def fixed_silence(
    audio: torch.Tensor,
    start: float,
    end: float,
    duration: float = 0.4,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """
    Add silence to the audio
    """

    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    audio_pre_silence = audio[:start_sample]
    audio_post_silence = audio[end_sample:]
    silence = torch.zeros(int(duration * sample_rate))
    return torch.cat((audio_pre_silence, silence, audio_post_silence))


def make_turn_hold(
    waveform: torch.Tensor,
    start: float,
    end: float,
    duration_scale: float = 1.5,
    intensity_scale: float = 2.0,
    pre_pitch_time: float = 0.5,
    apply_duration: bool = True,
    apply_pitch: bool = True,
    apply_intensity: bool = True,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
    f0_max: int = F0_MAX,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    assert waveform.ndim == 1, f"Expects (n_samples) got {tuple(waveform.shape)}"
    sound = torch_to_praat_sound(waveform, sample_rate)

    if apply_duration:
        sound = lengthening(
            sound,
            start,
            end,
            scale=duration_scale,
            hop_time=hop_time,
            f0_min=f0_min,
            f0_max=f0_max,
        )

    if apply_pitch:
        sound = flatten_pitch(
            sound,
            start,
            end,
            pre_pitch_time,
            hop_time=hop_time,
            f0_min=f0_min,
            f0_max=f0_max,
            sample_rate=sample_rate,
        )

    if apply_intensity:
        return raise_intensity(
            sound, start, end, intensity_scale, sample_rate=sample_rate
        )

    return praat_to_torch(sound)[0]


def first_sentence_hold(
    d,
    audio_path,
    tg_path,
    apply_pitch=True,
    apply_intensity=True,
    apply_duration=False,
    duration_scale=1.5,
    intensity_scale=2,
):
    tg = load_text_grid(tg_path)
    audio = load_audio(audio_path)[0, 0]  # mono
    times = find_hold_times(d, tg)

    hold_audio = make_turn_hold(
        audio,
        start=times["word"]["start"],
        end=times["word"]["end"],
        apply_duration=apply_duration,
        apply_pitch=apply_pitch,
        apply_intensity=apply_intensity,
        duration_scale=duration_scale,
        intensity_scale=intensity_scale,
    )
    return hold_audio


def plot_prosody(pitch, intensity, hop_time, ax, min_intensity=-200):
    """
    Plot pitch and intensity
    """
    pitch[pitch == 0] = torch.nan
    ax.plot(torch.arange(len(pitch)) * hop_time, pitch, "o", markersize=3, color="b")

    if len(intensity) < len(pitch):
        diff = len(pitch) - len(intensity)
        intensity = torch.cat((torch.tensor([min_intensity] * diff), intensity))

    int_ax = ax.twinx()
    # intensity[intensity < min_intensity] = torch.nan
    int_ax.plot(torch.arange(len(intensity)) * hop_time, intensity, color="r")
    int_ax.semilogy()
    return ax


def plot_audio(
    audio, pitch, intensity, sample_rate: int = 16000, hop_time: float = HOP_TIME
):
    """Plot audio, mel-spectrogram, pitch, and intensity"""
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    _ = plot_waveform(waveform=audio, ax=ax[0])
    plot_mel_spectrogram(
        y=audio.unsqueeze(0), ax=[ax[1]], sample_rate=sample_rate, hop_time=0.01
    )
    plot_prosody(pitch=pitch, intensity=intensity, hop_time=hop_time, ax=ax[2])

    for a in ax:
        a.set_xlim(0, audio.shape[-1] / sample_rate)
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    import sounddevice as sd
    from vap.plot_utils import plot_waveform, plot_mel_spectrogram

    root = "data/multiwoz_tts_utts_dev"
    df_path = join(root, "data_tts.pkl")
    df = pd.read_pickle(df_path)
    df = df[df.permutation == "original"]
    # df = df[df.tts == "MicrosoftTTS"]
    df = df[df.tts == "GoogleTTS"]
    d = df.iloc[0]

    tg = load_text_grid(join(root, d["tg_path"]))
    audio = load_audio(join(root, d["audio_path"]))[0, 0]  # mono
    f0 = get_single_pitch(audio)
    # intensity = get_single_intensity(audio)
    intensity = rms(audio)

    fig_orig, ax_orig = plot_audio(audio, f0, intensity)
    sd.play(audio, samplerate=SAMPLE_RATE)
    plt.pause(0.1)

    sound = torch_to_praat_sound(audio)
    times = find_hold_times(d, tg)

    hold_audio = fixed_silence(
        audio, start=times["silence"]["start"], end=times["silence"]["end"]
    )
    print("Hold     audio: ", len(hold_audio) / SAMPLE_RATE)

    hold_audio = make_turn_hold(
        audio,
        start=times["word"]["start"],
        end=times["word"]["end"],
        apply_pitch=True,
        apply_intensity=True,
        intensity_scale=4,
        duration_scale=1.5,
    )
    print("Original audio: ", len(audio) / SAMPLE_RATE)
    print("Hold     audio: ", len(hold_audio) / SAMPLE_RATE)

    hold_f0 = get_single_pitch(hold_audio)
    hold_intensity = get_single_intensity(hold_audio)
    print("hold_audio: ", tuple(hold_audio.shape))
    print("hold_f0: ", tuple(hold_f0.shape))
    print("hold_intensity: ", tuple(hold_intensity.shape))
    fig, ax = plot_audio(hold_audio, hold_f0, hold_intensity)
    sd.play(hold_audio, samplerate=SAMPLE_RATE)
    plt.pause(0.1)
