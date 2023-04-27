import re
from os.path import basename, exists, join
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
from vap.utils import write_txt

from vap_tts import AmazonTTS, GoogleTTS, MicrosoftTTS
from vap_tts.utils import extract_sentences

"""
Example paths

root/multiwoz_tts_utts_dev_100
├── alignment
│   ├── original
│   ├── comma    (TODO) 
│   ├── filler   (TODO)
│   └── prosody  (TODO)
└── audio
    ├── original
    ├── comma    (TODO)
    ├── filler   (TODO)
    └── prosody  (TODO)
"""

VALID_TTS = ["AmazonTTS", "GoogleTTS", "MicrosoftTTS"]

# TODO: MultiWoz, SSML (pitch, filled, pause, intensity)


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="data/multiwoz_tts_utts.pkl")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--tts", nargs="+", type=str, default=["AmazonTTS", "GoogleTTS", "MicrosoftTTS"]
    )
    parser.add_argument(
        "--permutations", nargs="+", type=str, default=["original", "comma", "filler"]
    )
    args = parser.parse_args()

    for tts in args.tts:
        if tts not in VALID_TTS:
            raise ValueError(f"Invalid TTS: {tts}. Valid TTS: {VALID_TTS}")

    # Data must be pickle format
    assert args.data.endswith(
        ".pkl"
    ), f"data_path must be a pickle file. Got {args.data}"

    return args


def create_paths(data_path: str, output_dir: str, permutations: List[str]):
    """
    Example:

    paths = {
        'root': 'data/multiwoz_tts_utts',
        'original': 'audio/original',
        'comma': 'audio/comma',
        'filler': 'audio/filler'
        }
    """
    paths = {}
    paths["root"] = join(output_dir, basename(data_path).replace(".pkl", ""))
    audio_dir = "audio"
    if args.dev:
        paths["root"] += "_dev"
    for p in permutations:
        rel_path = join(audio_dir, p)
        Path(join(paths["root"], rel_path)).mkdir(parents=True, exist_ok=True)
        paths[p] = rel_path
    return paths


def get_tts_class(tts_name: str):
    if tts_name.lower() == "amazontts":
        return AmazonTTS()
    elif tts_name.lower() == "googletts":
        return GoogleTTS()
    elif tts_name.lower() == "microsofttts":
        return MicrosoftTTS()
    else:
        raise ValueError(f"Invalid TTS: {tts_name}. Must be one of {VALID_TTS}")


# TODO: PERMUTATION
# 1. Commas instead of punctation delimits sentences
# 2. Inseart "fillers" (um, uh, etc.) and conjunction "so"
def replace_period_with_comma_simple(text: str):
    return re.sub(r"(\w)\.", "\1,", text)


def replace_period_with_filler_and_comma_simple(text: str, filler: str = "um"):
    return re.sub(r"\.", f" {filler},", text)


def replace_period_with_comma(text):
    """
    A simple regexp can be applied to change all periods to commas:
    new_text = re.sub(r'\.', ',', text)
    but we lowercase the first letter in the next sentence so becomes a litte more complicated...
    """
    sents = extract_sentences(text)
    new_text = ""
    last_is_comma = False
    for sent, punct in zip(sents["text"], sents["punctuation"]):
        if punct == ".":
            new_text += sent + ", "
            last_is_comma = True
        else:
            if last_is_comma:
                new_text += sent[0].lower() + sent[1:] + punct
            else:
                new_text += " " + sent + punct
            last_is_comma = False
    return new_text.strip()


def replace_period_with_filler_and_comma(text, filler: str = "um"):
    """
    A simple regexp can be applied to change all periods to commas:
    new_text = re.sub(r'\.', ',', text)
    but we lowercase the first letter in the next sentence so becomes a litte more complicated...
    """
    sents = extract_sentences(text)
    new_text = ""
    last_is_comma = False
    for sent, punct in zip(sents["text"], sents["punctuation"]):
        if punct == ".":
            new_text += sent + f" {filler}, "
            last_is_comma = True
        else:
            if last_is_comma:
                new_text += sent[0].lower() + sent[1:] + punct
            else:
                new_text += " " + sent + punct
            last_is_comma = False
    return new_text.strip()


def generate_tts_audio(df, args):
    paths = create_paths(args.data, args.output_dir, args.permutations)

    if len(args.tts) != 3:
        print("Warning! will overwrite data_utts.pkl without all TTS")
        input("Continue?")

    data = []
    for tts_name in args.tts:
        tts = get_tts_class(tts_name)
        tts_name = tts.__class__.__name__
        for perm in args.permutations:
            for index, row in tqdm(
                df.iterrows(), total=len(df), desc=tts_name + "-" + perm
            ):
                d = row.to_dict()
                sample_id = row.sample_id
                text = row.text

                if perm != "original":
                    if perm == "comma":
                        text = replace_period_with_comma(text)
                    elif perm == "filler":
                        text = replace_period_with_filler_and_comma(text)
                    sents = [s.strip() for s in text.split(",")]
                    d["text"] = text
                    d["sentences"] = sents
                    d["n_words"] = [len(s.split(" ")) for s in sents]
                rel_audio_path = join(paths[perm], f"{sample_id}_{tts_name}_{perm}.wav")
                d["permutation"] = perm
                d["tts"] = tts_name
                d["audio_path"] = rel_audio_path
                d["txt_path"] = rel_audio_path.replace(".wav", ".txt")
                d["tg_path"] = rel_audio_path.replace("audio/", "alignment/").replace(
                    ".wav", ".TextGrid"
                )
                wavpath = join(paths["root"], rel_audio_path)
                if args.overwrite or not exists(wavpath):
                    tts(text, wavpath)
                    write_txt([text], wavpath.replace(".wav", ".txt"))
                data.append(d)

    df_new = pd.DataFrame(data)
    df_path = join(paths["root"], "data_tts.pkl")
    df_new.to_pickle(df_path)
    print("Saved data -> ", df_path)


if __name__ == "__main__":

    args = get_args()

    # Load data
    df = pd.read_pickle(args.data)

    # use a smaller subset if dev=True
    if args.dev:
        df = df.iloc[:100]

    generate_tts_audio(df, args)
