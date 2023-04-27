from os.path import join
import matplotlib.pyplot as plt
import torch

from vap.plot_utils import plot_vap, plot_words_time
from vap_tts.utils import load_audio, load_text_grid, load_vap_output


def visualize_sample(d, root, post_silence: float = 2):
    if not isinstance(d, dict):
        d = d.to_dict()

    out = load_vap_output(join(root, d["vap_path"]))
    tg = load_text_grid(join(root, d["tg_path"]))
    audio = load_audio(join(root, d["audio_path"]), pad=post_silence)

    ######################
    fig, axs = plot_vap(
        audio[0],
        p_now=out["p_now"][0, :, 0],
        p_fut=out["p_future"][0, :, 0],
        vad=out["vad"][0],
        plot=False,
    )
    axs[-2].axvline(tg["words"]["ends"][-1], color="r")
    axs[-1].axvline(tg["words"]["ends"][-1], color="r")
    plot_words_time(
        words=tg["words"]["words"],
        starts=tg["words"]["starts"],
        ends=tg["words"]["ends"],
        ax=axs[2],
        fontsize=16,
    )
    plt.tight_layout()
    return fig, axs


def visualize_human_sample(d, post_silence: float = 0):
    if not isinstance(d, dict):
        d = d.to_dict()

    mono_out = load_vap_output(d["vap_path_mono"])
    audio = load_audio(d["audio_path"], pad=post_silence)

    fig, axs = plot_vap(
        audio[0],
        p_now=mono_out["p_now"][0, :, 0],
        p_fut=mono_out["p_future"][0, :, 0],
        vad=mono_out["vad"],
        plot=False,
    )
    t0 = d["start"]
    plot_words_time(
        words=d["words"],
        starts=torch.tensor(d["starts"]) - t0,
        ends=torch.tensor(d["ends"]) - t0,
        ax=axs[2],
        fontsize=16,
    )
    plt.tight_layout()
    return fig, axs
