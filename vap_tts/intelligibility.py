from argparse import ArgumentParser
from os.path import basename, join
from glob import glob
from tqdm import tqdm
import pandas as pd
import re

import whisper
from jiwer import wer

from vap.utils import read_txt

"""
Read all files
"""

TTS_NAMES = ["AmazonTTS", "GoogleTTS", "MicrosoftTTS"]


def remove_punctuation(text):
    """
    remove punctuation ",", ".", "!", "?", ";", ":" from text
    """
    return re.sub(r"[,.;:!?]", "", text)


def fix_punctuation_wer(idf):
    """
    Must remove punctuation in wer calculation because the added/removed punctuation
    interferes with the wer calculation.
    """
    new_data = []
    for i in tqdm(range(len(idf))):
        d = idf.iloc[i]
        t = remove_punctuation(d.text)
        at = remove_punctuation(d.asr_text)
        word_error_rate = wer(t, at)
        new_d = d.to_dict()
        new_d["wer_punct"] = word_error_rate
        new_data.append(new_d)
    df = pd.DataFrame(new_data)
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--root", type=str, default="data/paper")
    args = parser.parse_args()

    model = whisper.load_model(args.model)

    error = {"invalid_tts": []}
    data = []
    audio_root = join(args.root, "audio")
    for perm in ["original", "comma", "filler", "fsh"]:
        perm_root = join(audio_root, perm)
        for wav_path in tqdm(glob(join(perm_root, "*.wav")), desc=perm):
            filename = basename(wav_path).replace(".wav", "")
            sample_id, turn_id, _, tts, perm_name = filename.split("_")
            if tts not in TTS_NAMES:
                print("Invalid TTS: ", tts)
                error["invalid_tts"].append(wav_path)
                continue
            # Read original text
            txt_path = wav_path.replace(".wav", ".txt")
            text = read_txt(txt_path)[0].strip().lower()
            # Extract ASR text
            result = model.transcribe(wav_path)
            asr_text = result["text"].strip().lower()
            word_error_rate = wer(text, asr_text)
            entry = {
                "tts": tts,
                "perm": perm,
                "text": text,
                "sample_id": sample_id,
                "asr_text": asr_text,
                "wav_path": wav_path,
                "txt_path": txt_path,
                "wer": word_error_rate,
            }
            data.append(entry)
    df = pd.DataFrame(data)
    df.to_pickle(join(args.root, f"intelligibility_{args.model}.pkl"))
