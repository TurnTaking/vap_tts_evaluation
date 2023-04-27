from argparse import ArgumentParser
from os.path import basename, join
from glob import glob
from tqdm import tqdm
import pandas as pd
import json

from vap.utils import read_txt

"""
Read all files

Assume already ran mos script.
"""

TTS_NAMES = ["AmazonTTS", "GoogleTTS", "MicrosoftTTS", "ljs-fp-mel", "tcc-tt2"]

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--root", type=str, default="data/paper")
    parser.add_argument("--mos-json", type=str, default="predicted_mos.json", help="mos json file output from mos prediction script")
    args = parser.parse_args()

    error = {"invalid_tts": []}
    data = []
    audio_root = join(args.root, "audio")
    for perm in ["original", "comma", "filler", "fsh"]:
        perm_root = join(audio_root, perm)

        # read mos json
        mos_json = join(perm_root, args.mos_json)
        try:
            mos_dict = json.load(open(mos_json))
        except:
            raise RuntimeError("Error reading mos json file: ", mos_json)

        for wav_path in tqdm(glob(join(perm_root, "*.wav")), desc=perm):
            filename = basename(wav_path).replace(".wav", "")
            sample_id, turn_id, _, tts, perm_name = filename.split("_")
            if tts not in TTS_NAMES:
                print("Invalid TTS: ", tts)
                error["invalid_tts"].append(wav_path)
                continue
            
            # read mos
            wav_name = basename(wav_path)
            if wav_name not in mos_dict:
                raise RuntimeError("Error: mos not found for wav: ", wav_path)
            mos = mos_dict[wav_name]

            entry = {
                "tts": tts,
                "perm": perm,
                "sample_id": sample_id,
                "wav_path": wav_path,
                "mos": mos,
            }
            data.append(entry)
    df = pd.DataFrame(data)
    df.to_pickle(join(args.root, f"mos.pkl"))
