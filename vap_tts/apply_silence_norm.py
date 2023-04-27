from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
import pandas as pd

# import re
import shutil
import torchaudio
import torch

from vap.audio import load_waveform
from vap_tts.vadder import VadSilero
from vap_tts.utils import load_text_grid

"""
Create a new folder with silence normalized audio files.
"""


def find_tg_pause_trigram(d, root) -> Tuple[float, float]:
    """
    Using the word-prior and word-post of the pause to match silences
    detected by forced aligner
    """

    # TextGrid
    tg = load_text_grid(join(root, d["tg_path"]))
    tw = tg["words"]

    # Find trigram (excluding the silence)
    a = d.sentences[0].lower().split(" ")
    b = d.sentences[1].lower().split(" ")
    word_prior = a[-1]
    word_post = b[0]

    # find silences by tg
    pause_start, pause_end = -1.0, -1.0
    for ii, (w, s, e) in enumerate(
        zip(tw["words"][:-1], tw["starts"][:-1], tw["ends"][:-1])
    ):
        if w == "":
            prev_word = tw["words"][ii - 1]
            next_word = tw["words"][ii + 1]
            if prev_word == word_prior and next_word == word_post:
                pause_start = float(s)
                pause_end = float(e)
                break
    return pause_start, pause_end


def fix_silence(
    audio: torch.Tensor,
    start: float,
    end: float,
    pause_duration: float = 0.4,
    sample_rate: int = 16_000,
) -> torch.Tensor:
    """
    Add silence to the audio
    """
    assert audio.ndim == 1, f"Audio must be 1D, got {audio.ndim}"

    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    pause_samples = int(pause_duration * sample_rate)

    audio_pre_silence = audio[:start_sample]
    audio_post_silence = audio[end_sample:]
    silence = torch.zeros(pause_samples).float()
    return torch.cat((audio_pre_silence, silence, audio_post_silence))


class Silencer:
    def __init__(self, root, new_root, pause_duration=0.4, overwrite=False):
        self.root = root
        self.new_root = new_root
        self.pause_duration = pause_duration
        self.overwrite = overwrite
        self.sample_rate = 16_000

        self.df_path = join(root, "data_tts.pkl")
        self.df = pd.read_pickle(self.df_path)

        self.vadder = VadSilero()

    def transform(self):
        # Create new folder
        Path(self.new_root).mkdir(parents=True, exist_ok=True)

        # Iterate over samples and normalize silence and copy txt
        error = {"exists": 0, "no_explicit_pause": 0, "samples": []}
        perm_diffs = {"original": [], "comma": [], "filler": []}
        data = []
        for ii in tqdm(range(len(self.df)), desc="Silence Norm"):
            d = self.df.iloc[ii]

            pause_start, pause_end = find_tg_pause_trigram(d, self.root)
            if pause_start < 0 and pause_end < 0:
                error["no_explicit_pause"] += 1
                error["samples"].append(d)
                continue

            audio, _ = load_waveform(
                join(self.root, d.audio_path), sample_rate=self.sample_rate
            )
            audio = audio[0]  # (1, n_samples) -> (n_samples,)

            n_orig = len(audio)
            silence_audio = fix_silence(
                audio,
                start=pause_start,
                end=pause_end,
                pause_duration=self.pause_duration,
            )
            n_sil = len(silence_audio)

            perm_diffs[d.permutation].append(n_sil - n_orig)

            # Save silence audio and copy corresponding text
            # Create audio-folder e.g. data/paper/audio/original
            Path(join(self.new_root, "audio", d.permutation)).mkdir(
                parents=True, exist_ok=True
            )
            torchaudio.save(
                join(self.new_root, d.audio_path),
                silence_audio.unsqueeze(0),
                sample_rate=self.sample_rate,
            )
            shutil.copyfile(
                src=join(self.root, d.txt_path), dst=join(self.new_root, d.txt_path)
            )
            new_d = d.to_dict()
            data.append(new_d)

        for k, v in error.items():
            print(f"{k}: {v}")

        df = pd.DataFrame(data)
        pd_path = join(self.new_root, "data_tts.pkl")
        df.to_pickle(pd_path)
        print("Saved silence examples -> ", pd_path)
        return df, error, perm_diffs


def checkhealth():

    import matplotlib.pyplot as plt
    import sounddevice as sd

    # Orginal
    orig_path = "data/multiwoz_tts_utts/data_tts.pkl"
    df = pd.read_pickle(orig_path)
    print(df)

    new_root = "data/paper"
    sdf = pd.read_pickle(join(new_root, "data_tts.pkl"))
    print(sdf)

    # Check silences
    vadder = VadSilero()
    ipu_pause = {"no_vad": 0, "single": 0, "pair": [], "more": 0}
    for ii in tqdm(range(len(df))):
        d = df.iloc[ii]
        audio_path = join(new_root, d.audio_path)
        x = load_waveform(audio_path)[0][0]
        vad_list = vadder.vad_list(x)
        if len(vad_list) == 0:
            ipu_pause["no_vad"] += 1
        elif len(vad_list) == 1:
            ipu_pause["single"] += 1
        elif len(vad_list) == 2:
            pause_dur = vad_list[1][0] - vad_list[0][1]
            ipu_pause["pair"].append(pause_dur)
        else:
            ipu_pause["more"] += 1

    print("no_vad: ", ipu_pause["no_vad"])
    print("single: ", ipu_pause["single"])
    print("pair: ", len(ipu_pause["pair"]))
    print("more: ", ipu_pause["more"])
    plt.hist(ipu_pause["pair"])
    plt.show()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/multiwoz_tts_utts")
    parser.add_argument("--new_root", type=str, default="./data/paper")
    parser.add_argument("--pause_duration", type=float, default=0.4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    silencer = Silencer(args.root, args.new_root, args.pause_duration, args.overwrite)
    df, error, perm_diffs = silencer.transform()
