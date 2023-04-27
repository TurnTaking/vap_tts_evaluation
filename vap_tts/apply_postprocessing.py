from argparse import ArgumentParser
import pandas as pd
from os.path import dirname, join
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
from typing import Tuple
import shutil


from vap.audio import load_waveform

from vap_tts.postprocessing import make_turn_hold
from vap_tts.utils import load_text_grid


def find_vad_pause(audio, pe, vadder):
    vad_list = torch.tensor(vadder.vad_list(audio))
    dist = vad_list[:, 0] - pe
    idx = torch.arange(len(dist))
    idx = idx[dist > 0]
    dist = dist[dist > 0]
    choose = dist.argmin()
    chosen = idx[choose]
    # return the start of the closest vad segment
    return vad_list[chosen][0].item()


# taken from apply_silence_norm.py
def find_tg_pause_word_trigram(d, root) -> Tuple[float, float]:
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
    word_start, word_end = -1.0, -1.0
    for ii, w in enumerate(tw["words"][:-1]):
        if w == "":
            prev_word = tw["words"][ii - 1]
            next_word = tw["words"][ii + 1]
            if prev_word == word_prior and next_word == word_post:
                word_start = float(tw["starts"][ii - 1])
                word_end = float(tw["ends"][ii - 1])
                break
    return word_start, word_end


class PostProcessor:
    def __init__(self, root, intensity_scale=1.5, duration_scale=1.5):
        self.root = root
        self.df_path = join(args.root, "data_tts.pkl")
        self.df = pd.read_pickle(self.df_path)
        self.sample_rate = 16_000

        self.intensity_scale = intensity_scale
        self.duration_scale = duration_scale

    def process(self):
        # Only care about originals
        pdf = self.df[self.df.permutation == "original"]
        data = []
        error = {"word": []}
        for _, d in tqdm(pdf.iterrows(), total=len(pdf), desc="PostProcessing hold"):
            ws, we = find_tg_pause_word_trigram(d, self.root)
            if ws < 0 or we < 0:
                error["word"].append(d)
                continue

            audio = load_waveform(join(self.root, d["audio_path"]))[0][0]
            hold_audio = make_turn_hold(
                audio,
                start=ws,
                end=we,
                apply_duration=True,
                apply_pitch=True,
                apply_intensity=True,
                duration_scale=self.duration_scale,
                intensity_scale=self.intensity_scale,
            )

            # Change relevant values
            new_d = d.to_dict()
            new_d["permutation"] = d.permutation.replace("original", "fsh")
            new_d["audio_path"] = new_d["audio_path"].replace("original", "fsh")
            new_d["tg_path"] = new_d["tg_path"].replace("original", "fsh")
            new_d["txt_path"] = new_d["txt_path"].replace("original", "fsh")

            # Create subfolder if it does not exist
            dir_name = dirname(join(self.root, new_d["audio_path"]))
            Path(dir_name).mkdir(parents=True, exist_ok=True)

            # Save silence audio and copy corresponding text
            torchaudio.save(
                join(self.root, new_d["audio_path"]),
                hold_audio.unsqueeze(0),
                sample_rate=self.sample_rate,
            )
            shutil.copyfile(
                src=join(self.root, d.txt_path), dst=join(self.root, new_d["txt_path"])
            )
            data.append(new_d)

        print("Postprocessed: ", len(data), "/", len(pdf))
        print("Did not find word: ", len(error["word"]))

        data = data + [d.to_dict() for _, d in self.df.iterrows()]
        new_df = pd.DataFrame(data)
        print("New total entries: ", len(new_df))
        pd_path = join(args.root, "data_post_proc.pkl")
        new_df.to_pickle(pd_path)
        print("Saved postprocessing -> ", pd_path)
        return new_df, error


def debug():
    from vap_tts.vadder import VadSilero
    from vap_tts.apply_silence_norm import find_tg_pause_trigram
    from vap.plot_utils import plot_waveform, plot_mel_spectrogram
    import matplotlib.pyplot as plt

    root = "data/paper"
    pd_path = join(root, "data_post_proc.pkl")
    pdf = pd.read_pickle(pd_path)
    # pdf = pdf[pdf.permutation == "fsh"]
    print(pdf)

    vadder = VadSilero()
    diffs = []
    for i in tqdm(range(len(pdf))):
        d = pdf.iloc[i]
        ps, pe = find_tg_pause_trigram(d, root)
        diff = pe - ps
        if diff < 0.3:
            audio = load_waveform(join(root, d["audio_path"]))[0][0]
            pe = find_vad_pause(audio, pe, vadder)
            diff = pe - ps
        diffs.append(diff)
    plt.hist(diffs, bins=20)
    plt.show()

    # fig, ax = plt.subplots(2, 1)
    # plot_waveform(audio[0], ax=ax[0])
    # plot_mel_spectrogram(audio, ax=[ax[1]])
    # for a in ax:
    #     a.axvline(ps, color="r")
    #     a.axvline(pe, color="r")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/paper")
    parser.add_argument("--duration_scale", type=float, default=1.5)
    parser.add_argument("--intensity_scale", type=float, default=1.5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    processor = PostProcessor(args.root, args.intensity_scale, args.duration_scale)
    df, error = processor.process()
