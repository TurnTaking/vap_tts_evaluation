from argparse import ArgumentParser
from os.path import dirname, exists, join
from pathlib import Path
import pandas as pd

from tqdm import tqdm

from vap_tts.utils import load_audio, load_model, save_output_json


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/paper")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./data/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
    )
    parser.add_argument(
        "--post_silence",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def debug():
    df = pd.read_pickle("data/paper/data_vap.pkl")
    print(df)
    print(df.permutation.unique())


if __name__ == "__main__":

    args = get_args()

    # Load dataframe
    pd_path = join(args.root, "data_post_proc.pkl")
    df = pd.read_pickle(pd_path)

    # Load model
    model, device = load_model(args.checkpoint)

    data = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"VAP: {args.root}"):
        d = row.to_dict()
        d["vap_path"] = d["audio_path"].replace("audio", "vap").replace(".wav", ".json")
        out_path = join(args.root, d["vap_path"])
        if args.overwrite or not exists(out_path):
            # Create dir if not exists
            Path(dirname(out_path)).mkdir(parents=True, exist_ok=True)
            # Load audio
            audio = load_audio(join(args.root, d["audio_path"]), pad=args.post_silence)
            # Run VAP
            out = model.probs(audio.to(device))
            # Save output
            save_output_json(out, out_path)
        data.append(d)

    # Save dataframe
    new_df = pd.DataFrame(data)
    pd_path = join(args.root, "data_vap.pkl")
    new_df.to_pickle(pd_path)
    print("Save new df -> ", pd_path)
