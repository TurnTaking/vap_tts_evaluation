import torch
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pyplot as plt

from vap.audio import load_waveform
from vap.plot_utils import plot_waveform, plot_mel_spectrogram

from vap_tts.vadder import VadSilero
from vap_tts.postprocessing import find_hold_times
from vap_tts.utils import load_text_grid, load_vap_output, TG
from vap_tts.apply_silence_norm import find_tg_pause_trigram


if __name__ == "__main__":

    root = "data/paper"
    target_silence = 0.4
    allow_pad = 0.1

    df0 = pd.read_pickle(join(root, "data_tts.pkl"))
    print(df0)

    vadder = VadSilero()

    def extract_last_silence(audio, pe):
        vad_list = torch.tensor(vadder.vad_list(audio))
        dist = vad_list[:, 0] - pe
        idx = torch.arange(len(dist))
        idx = idx[dist > 0]
        dist = dist[dist > 0]
        choose = dist.argmin()
        chosen = idx[choose]
        # return the start of the closest vad segment
        return vad_list[chosen][0].item()

    other = []
    good = []
    for i in tqdm(range(len(df0))):
        d = df0.iloc[i]
        tg = load_text_grid(join(root, d.tg_path))
        ps, pe = find_tg_pause_trigram(d, root)
        diff = pe - ps
        new_d = d.to_dict()
        if 0.3 <= diff <= 0.52:
            new_d["pause_start"] = ps
            new_d["pause_end"] = pe
            new_d["pause_dur"] = diff
            good.append(new_d)
        else:
            # Find silence using help from VAD
            audio = load_waveform(join(root, d.audio_path))[0][0]
            se = extract_last_silence(audio, pe)
            diff = se - pe
            new_d["pause_start"] = ps
            new_d["pause_end"] = se
            new_d["pause_dur"] = diff
            if 0.3 <= diff <= 0.52:
                good.append(new_d)
            else:
                other.append(new_d)
    print("Good: ", len(good))
    print("Other: ", len(other))
    odf = pd.DataFrame(other)
    gdf = pd.DataFrame(good)

    print("other max: ", odf.pause_dur.max())
    print("other min: ", odf.pause_dur.min())
    print("good max: ", gdf.pause_dur.max())
    print("good min: ", gdf.pause_dur.min())

    gdf.pause_dur.hist(bins=10)
    plt.show()

    silence = {"ok": [], "short": [], "long": []}
    for ii in tqdm(range(len(sdf))):
        d = sdf.iloc[ii]
        tg = load_text_grid(join(root, d.tg_path))
        ps, pe = find_tg_pause_trigram(d, root)
        audio = load_waveform(join(root, d.audio_path))[0][0]
        se = extract_last_silence(audio, pe)
        diff = se - pe
        if 0.3 <= diff <= 0.5:
            silence["ok"].append(diff)
        else:
            if diff < 0.3:
                silence["short"].append(diff)
            else:
                silence["long"].append(diff)
            fig, ax = plt.subplots(2, 1, sharex=True)
            plot_waveform(audio, ax=ax[0])
            plot_mel_spectrogram(audio.unsqueeze(0), ax=[ax[1]])
            print(diff)
            for a in ax:
                # a.axvline(ps, color="b")
                a.axvline(pe, color="b", linewidth=2)
                a.axvline(se, color="r", linewidth=2)
            plt.tight_layout()
            plt.show()

    print("ok: ", len(silence["ok"]))
    print("short: ", len(silence["short"]))
    print("long: ", len(silence["long"]))

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].hist(silence["ok"], bins=20)
    ax[1].hist(silence["long"], bins=20)
    ax[2].hist(silence["short"], bins=20)
    plt.show()
