from argparse import ArgumentParser
from os.path import join
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import norm

from vap.audio import load_waveform
from vap.plot_utils import plot_waveform, plot_mel_spectrogram

from vap_tts.vadder import VadSilero
from vap_tts.apply_silence_norm import find_tg_pause_trigram
from vap_tts.apply_postprocessing import find_vad_pause
from vap_tts.utils import load_text_grid, load_vap_output

FRAME_HZ: int = 50
COMPLETION_POINTS: List[str] = ["FIRST", "LAST"]

"""
Results

* Shift-Volume p-now, p-future
    * Calculated w.r.t the 50% cutoff
    * Positive P-now volume -> Invites listener to speak
    * Positive P-future volume -> Invites listener to "take turn"
    * Both are positive -> Strong turn-yielding signal
    * Only P-now is positive -> Ambiguous signal
"""


def get_shift_times(tg, post_silence) -> Tuple[float, float]:
    end = tg["words"]["ends"][-1]
    return end, end + post_silence


def get_volume(p, start, end, frame_hz: int = 50):
    ps = round(frame_hz * start)
    pe = round(frame_hz * end)
    return (p[..., ps:pe, 1] - 0.5).mean().item()


class Results:
    def __init__(
        self,
        root="data/paper",
        post_silence=1.0,
        target_silence=0.4,
        allow_pad=0.1,
        min_detected_silence=0.3,
    ):
        self.root = root
        self.allow_pad = allow_pad
        self.post_silence = post_silence
        self.target_silence = target_silence
        self.min_detected_silence = min_detected_silence

        self.silence_pad_frame_time = 0.02

        self.df_path = join(root, "data_vap.pkl")
        self.df: pd.DataFrame = pd.read_pickle(self.df_path)

        # Vad
        self.vadder = VadSilero()

    def find_silence_with_vad(self, d, pause_start, pause_end):
        """
        Forced aligner makes errors, so we need to find the silence
        in a more controlled/custom way.

        The forced aligner thinks that the pause ends where it actually starts...



        1. Find the start of the last word of the first sentence.
        2. detect silence after that (to avoid silences in first utterance)
        """

        x, _ = load_waveform(join(self.root, d.audio_path))

        vad_list = self.vadder.vad_list(x[0])

        # pause_start, pause_end = -1, -1
        sil_start, sil_end = -1, -1
        if len(vad_list) == 2:
            sil_start = vad_list[0][-1]
            sil_end = vad_list[-1][0]
        else:
            print("vad longer than 2 IPUS")
            return -1, -1

        # round(sil_end - pause_end, 2)

        new_pause_start = -1
        new_pause_end = -1
        if self.min_detected_silence <= sil_end - pause_end:
            new_pause_start = pause_end
            new_pause_end = sil_end
            # sd.play(x[0], samplerate=16_000)
        else:
            fig, ax = plt.subplots(2, 1, sharex=True)
            plot_waveform(x[0], ax=ax[0])
            plot_mel_spectrogram(x, ax=[ax[1]])
            for a in ax:
                a.axvline(pause_start, color="b")
                a.axvline(pause_end, color="b")
                a.axvline(sil_start, color="r")
                a.axvline(sil_end, color="r")
            print("Bad times: ", round(pause_end - pause_start, 2))
            print(d.sample_id, d.permutation)
            plt.show()
        return new_pause_start, new_pause_end

    def vap_metrics(self, save=False, early_yield_time=0.6):
        skipped = {"n": 0, "start_ends": [], "samples": [], "vad_fix": 0}
        data = []
        for ii in tqdm(range(len(self.df))):
            d = self.df.iloc[ii]
            out = load_vap_output(join(self.root, d.vap_path))

            ps, pe = find_tg_pause_trigram(d, self.root)
            diff = pe - ps

            # Extracted pause it shorter than it should be
            # -> We load the audio and use the VAD to find the pause
            if diff < 0.3:
                audio = load_waveform(join(self.root, d["audio_path"]))[0][0]
                pe = find_vad_pause(audio, pe, self.vadder)
                diff = pe - ps

            # Skip files where forced-alignment have failed
            if not (0.3 <= diff <= 0.52):
                skipped["start_ends"].append((ps, pe))
                skipped["samples"].append(d)
                skipped["n"] += 1
                continue

            # Add a single frame as pad to avoid a frame with actual activity
            ps += self.silence_pad_frame_time
            pe -= self.silence_pad_frame_time

            # Get pause-region values
            vnp = get_volume(out["p_now"], ps, pe)
            vfp = get_volume(out["p_future"], ps, pe)

            # Get end-region averages
            tg = load_text_grid(join(self.root, d["tg_path"]))
            shift_start, shift_end = get_shift_times(tg, post_silence=self.post_silence)

            vns = get_volume(out["p_now"], shift_start, shift_end)
            vfs = get_volume(out["p_future"], shift_start, shift_end)

            # predictive
            pred_now_start = shift_start - early_yield_time
            eyn = get_volume(out["p_now"], pred_now_start, shift_start)
            eyf = get_volume(out["p_future"], pred_now_start, shift_start)
            eyff = get_volume(out["p_future"], shift_start - 1.3, shift_start)

            # Add data
            new_d = d.to_dict()
            new_d["pause_now"] = vnp
            new_d["pause_fut"] = vfp
            new_d["late_yield_now"] = vns
            new_d["late_yield_fut"] = vfs
            new_d["early_yield_now"] = eyn
            new_d["early_yield_fut"] = eyf
            new_d["pred_early_yield"] = eyff
            new_d["pause_duration"] = diff
            data.append(new_d)
        print("Skipped: ", skipped["n"])
        df = pd.DataFrame(data)
        if save:
            df_path = join(self.root, "results.pkl")
            df.to_pickle(df_path)
            print("Saved results -> ", df_path)
        return df, skipped

    @staticmethod
    def classify(pn, pf):
        # Get sign -1, 1
        pn_sign = pn.sign()
        pf_sign = pf.sign()

        # Where are they the same?
        same = pn_sign == pf_sign
        diff = torch.logical_not(same)

        # Where is p_now positive/negative
        ispos = torch.where(pn_sign > 0)
        isneg = torch.where(pn_sign < 0)

        # sum
        N = len(same)
        n_shift = same[ispos].sum().item() / N
        n_hold = same[isneg].sum().item() / N
        n_weak_shift = diff[ispos].sum().item() / N
        return {
            "shift": n_shift,
            "weak_shift": n_weak_shift,
            "hold": n_hold,
        }

    @staticmethod
    def classify_early_yield(pn, pf):
        # Get sign -1, 1
        pn_sign = pn.sign()
        pf_sign = pf.sign()

        # Where are they the same?
        same = pn_sign == pf_sign
        diff = torch.logical_not(same)

        # Where is p_now positive/negative
        ispos = torch.where(pf_sign > 0)
        # isneg = torch.where(pf_sign < 0)

        # sum
        N = len(same)
        n_pred_shift_strong = same[ispos].sum().item() / N
        n_pred_shift_weak = diff[ispos].sum().item() / N
        return {
            "early_yield_strong": n_pred_shift_strong,
            "early_yield_weak": n_pred_shift_weak,
        }

    @staticmethod
    def calculate_stats(rdf, tts, perm):
        """
        Calculates stats for 'weak-shift', 'strong-shift' and 'strong-hold'
        weak-shift:    p_now -> shift, p_fut -> hold
        strong-shift:  p_now -> shift, p_fut -> shift
        strong-hold:   p_now -> hold, p_fut -> hold
        """

        df = rdf[rdf.tts == tts]
        df = df[df.permutation == perm]

        # Pause region
        pn = torch.tensor(df["pause_now"].to_list())
        pf = torch.tensor(df["pause_fut"].to_list())
        p_region = Results.classify(pn, pf)

        # End region
        pn = torch.tensor(df["late_yield_now"].to_list())
        pf = torch.tensor(df["late_yield_fut"].to_list())
        e_region = Results.classify(pn, pf)

        # Early yield region

        pn = torch.tensor(df["early_yield_now"].to_list())
        pf = torch.tensor(df["early_yield_fut"].to_list())
        ey = Results.classify_early_yield(pn, pf)

        return {
            "hold_weak": p_region["weak_shift"] + p_region["hold"],
            "hold_strong": p_region["hold"],
            "late_yield_strong": e_region["shift"],
            "late_yield_weak": e_region["shift"] + e_region["weak_shift"],
            "early_yield_strong": ey["early_yield_strong"],
            "early_yield_weak": ey["early_yield_weak"],
        }

    @staticmethod
    def get_results(rdf):
        stats = {}
        for tts in rdf.tts.unique():
            stats[tts] = {}
            for perm in rdf.permutation.unique():
                stats[tts][perm] = Results.calculate_stats(rdf, tts, perm)
        return stats

    @staticmethod
    def plot_histograms(df):
        g = df.groupby(["permutation"])
        c = ["r", "g", "b", "orange"]
        kwargs = {"bins": 50, "density": True, "range": [-0.5, 0.5], "alpha": 0.5}
        figs = {}
        for prob_type in ["pause_now", "pause_fut", "shift_now", "shift_fut"]:
            fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True)
            for ii, label in enumerate(["original", "comma", "filler", "fsh"]):
                v = g.get_group(label)[prob_type]
                v.hist(label=label, ax=ax[ii], color=c[ii], **kwargs)
                ax[ii].axvline(0, color="k", linestyle="dashed")
                m = torch.tensor(v.to_list()).mean()
                s = torch.tensor(v.to_list()).std()
                x = torch.linspace(-0.5, 0.5, 100)
                y = norm.pdf(x, m, s)
                ax[ii].plot(x, y, color="k", linewidth=2)
                ax[ii].legend(fontsize=14)
            ax[0].set_title(prob_type)
            plt.tight_layout()
            figs[prob_type] = fig
        return figs

    @staticmethod
    def plot_tts_histograms(df, tts="AmazonTTS", save_dir=None):
        g = df.groupby(["permutation", "tts"])
        c = ["r", "g", "b", "orange"]
        kwargs = {"bins": 50, "density": True, "range": [-0.5, 0.5], "alpha": 0.5}
        figs = {}
        for prob_type in ["pause_now", "pause_fut", "shift_now", "shift_fut"]:
            fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True)
            for ii, label in enumerate(["original", "comma", "filler", "fsh"]):
                v = g.get_group((label, tts))[prob_type]
                v.hist(label=label, ax=ax[ii], color=c[ii], **kwargs)
                ax[ii].axvline(0, color="k", linestyle="dashed")
                m = torch.tensor(v.to_list()).mean()
                s = torch.tensor(v.to_list()).std()
                x = torch.linspace(-0.5, 0.5, 100)
                y = norm.pdf(x, m, s)
                ax[ii].plot(x, y, color="k", linewidth=2)
                ax[ii].legend(fontsize=14)
            ax[0].set_title(prob_type)
            plt.tight_layout()
            if save_dir:
                spath = join(save_dir, f"{prob_type}_hist_{tts}.png")
                fig.savefig(spath)
                print("Saved -> ", spath)
            figs[prob_type] = fig
        return figs

    @staticmethod
    def plot_shift_wshift_hold(stats):
        fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(12, 8))
        w = 0.25
        for ax_idx, perm in enumerate(["original", "comma", "filler", "fsh"]):
            for ii, tts in enumerate(stats.keys()):
                s = stats[tts][perm]
                label = [None, None, None]
                if ax_idx == 0 and ii == 0:
                    label = ["Shift", "Weak-shift", "Hold"]
                ax[ax_idx].bar(
                    2 * ii - w,
                    s["shift"],
                    width=w,
                    color="r",
                    alpha=0.5,
                    label=label[0],
                )
                ax[ax_idx].bar(
                    2 * ii,
                    s["weak_shift"],
                    width=w,
                    color="b",
                    alpha=0.5,
                    label=label[1],
                )
                ax[ax_idx].bar(
                    2 * ii + w, s["hold"], width=w, color="g", alpha=0.5, label=label[2]
                )
            if ax_idx == 0:
                ax[ax_idx].legend()
        ax[-1].set_ylim([0, 1])
        ax[-1].set_xticks([0, 2, 4])
        ax[-1].set_xticklabels(list(stats.keys()))
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_pause_duration_hist(df):
        fig, ax = plt.subplots(1, 1)
        df["pause_duration"].hist(ax=ax, bins=50, density=True)
        plt.tight_layout()
        return fig, ax


def checkhealth():
    from vap_tts.plot_utils import visualize_sample

    root = "data/paper"
    rdf = pd.read_pickle(join(root, "results.pkl"))

    d = rdf.iloc[0]

    g = rdf.groupby(["permutation", "tts", "sample_id"])

    d = g.get_group(("original", "AmazonTTS", "SNG1013_5_SYSTEM")).iloc[0]
    # d = g.get_group(("original", "MicrosoftTTS", 'SNG1013_5_SYSTEM')).iloc[0]
    # d = g.get_group(("original", "GoogleTTS", 'SNG1013_5_SYSTEM')).iloc[0]
    pause_lims = find_tg_pause_trigram(d, root)
    fig, ax = visualize_sample(
        d, root, post_silence=1, vad=False, pause_lims=pause_lims
    )
    plt.show()

    resulter = Results()
    stats = resulter.get_results(rdf)

    fig, ax = resulter.plot_shift_wshift_hold(stats)
    plt.show()

    figs = resulter.plot_histograms(rdf)
    plt.show()

    figs = resulter.plot_tts_histograms(
        rdf, "MicrosoftTTS", save_dir="data/results/microsoft"
    )
    plt.close("all")
    figs = resulter.plot_tts_histograms(
        rdf, "GoogleTTS", save_dir="data/results/google"
    )
    plt.close("all")
    figs = resulter.plot_tts_histograms(
        rdf, "AmazonTTS", save_dir="data/results/amazon"
    )
    plt.close("all")

    # figs = resulter.plot_tts_histograms(rdf, "AmazonTTS")
    plt.show()

    for i in range(len(rdf)):
        d = rdf.iloc[i]
        p = load_text_grid(join(root, d.tg_path))["phones"]
        break


def print_table(stats, perm="original"):
    metrics = [
        "hold_weak",
        "hold_strong",
        "early_yield_weak",
        # "early_yield_strong",
        # "late_yield_weak",
        "late_yield_strong",
    ]
    print()
    print(perm.upper())
    top = "metric | " + " | ".join(list(stats.keys()))
    print(top)
    for met in metrics:
        row = met + " | "
        for tts in stats.keys():
            row += f"{stats[tts][perm][met]:.2f} | "
        print(row)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="data/paper")
    args = parser.parse_args()

    resulter = Results(root=args.root)

    rdf, skipped = resulter.vap_metrics(save=False, early_yield_time=0.4)

    stats = Results.get_results(rdf)

    print_table(stats, "original")
    print_table(stats, "comma")
    print_table(stats, "filler")
    print_table(stats, "fsh")

    # ORIGINAL
    # metric             | AmazonTTS | GoogleTTS | MicrosoftTTS
    # hold_weak          | 0.87*     | 0.79      | 0.86 |
    # hold_strong        | 0.31*     | 0.21      | 0.20 |
    # early_yield_weak   | 0.44*     | 0.28      | 0.32 |
    # late_yield_strong  | 0.95*     | 0.95*     | 0.90 |

    # COMMA
    # metric             | AmazonTTS | GoogleTTS | MicrosoftTTS
    # hold_weak          | 0.92*     | 0.83      | 0.90 |
    # hold_strong        | 0.48*     | 0.29      | 0.33 |
    # early_yield_weak   | 0.39*     | 0.28      | 0.28 |
    # late_yield_strong  | 0.92      | 0.94*     | 0.89 |

    # FILLER
    # metric             | AmazonTTS | GoogleTTS | MicrosoftTTS
    # hold_weak          | 0.93      | 0.77      | 1.00* |
    # hold_strong        | 0.57      | 0.26      | 1.00* |
    # early_yield_weak   | 0.38*     | 0.27      | 0.23  |
    # late_yield_strong  | 0.92      | 0.95*     | 0.88  |

    # FSH
    # metric             | AmazonTTS | GoogleTTS | MicrosoftTTS
    # hold_weak          | 0.97*     | 0.92      | 0.96  |
    # hold_strong        | 0.70      | 0.53      | 0.81* |
    # early_yield_weak   | 0.46*     | 0.26      | 0.29  |
    # late_yield_strong  | 0.95      | 0.97*     | 0.95  |

    # original
    # metric | ljs
    # hold_weak | 0.84 |
    # hold_strong | 0.30 |
    # early_yield_weak | 0.05 |
    # early_yield_strong | 0.00 |
    # late_yield_weak | 0.52 |
    # late_yield_strong | 0.39 |
    # -----------------
    # comma
    # metric | ljs
    # hold_weak | 0.90 |
    # hold_strong | 0.54 |
    # early_yield_weak | 0.06 |
    # early_yield_strong | 0.00 |
    # late_yield_weak | 0.56 |
    # late_yield_strong | 0.43 |
    # -----------------
    # filler
    # metric | ljs
    # hold_weak | 0.92 |
    # hold_strong | 0.69 |
    # early_yield_weak | 0.06 |
    # early_yield_strong | 0.00 |
    # late_yield_weak | 0.55 |
    # late_yield_strong | 0.38 |
    # -----------------
    # fsh
    # metric | ljs
    # hold_weak | 0.97 |
    # hold_strong | 0.85 |
    # early_yield_weak | 0.06 |
    # early_yield_strong | 0.00 |
    # late_yield_weak | 0.57 |
    # late_yield_strong | 0.42 |
    # -----------------
    # prosodyctrl
    # metric | ljs
    # hold_weak | 0.93 |
    # hold_strong | 0.54 |
    # early_yield_weak | 0.06 |
    # early_yield_strong | 0.00 |
    # late_yield_weak | 0.54 |
    # late_yield_strong | 0.40 |
    # -----------------
    # whisper-base intelligibility:
    # original 0.08052423752504483
    # comma 0.12727817548780837
    # filler 0.13976222055026
    # fsh 0.07788792276717735
    # prosodyctrl 0.07546514276172303
    # -----------------
    # whisper-large intelligibility:
    # original 0.07203284856329341
    # comma 0.12233415258819785
    # filler 0.14819521928608081
    # fsh 0.052282766473702086
    # prosodyctrl 0.06665242807360727
    # -----------------
    # predicted mos:
    # original 4.364327165852772
    # comma 4.350156627869457
    # filler 4.288354656395245
    # fsh 4.043473308006029
    # prosodyctrl 4.306674926452583
