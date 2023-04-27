from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import torch
import torchaudio
import re
from pathlib import Path
from os.path import exists, join
import matplotlib.pyplot as plt

from vap_dataset.corpus.switchboard import SwbReader
from vap.audio import load_waveform
from vap.utils import (
    get_vad_list_subset,
    vad_list_to_onehot,
    find_island_idx_len,
    write_txt,
)
from vap.plot_utils import plot_vap

from vap_tts.utils import (
    load_model,
    load_audio,
    save_output_json,
    load_text_grid,
    load_vap_output,
)
from vap_tts.results import get_volume

# from vap_tts.postprocessing import fixed_silence
# from vap.plot_utils import plot_vap

VALID_QS = ["qy", "qo", "qw"]
INVALID_PREVS = ["bh", "bk"]
# QY: Yes-No-Question, QO: Open-Question, QW: Wh-Question


def get_question_start_idx(das):
    ret = 0
    for idx, da in enumerate(das):
        # some 'nan' interpreted as float
        if isinstance(da, float):
            continue
        if da.startswith("q"):
            ret = idx
            break
    return ret


def get_unique_da_ordered(da):
    # sanity: nan -> str
    # gets interpreted as float
    da = [str(f) for f in da]

    das, ndas = [da[0]], [1]
    for da in da[1:]:
        if da != das[-1]:
            das.append(da)
            ndas.append(1)
        else:
            ndas[-1] += 1
    return das, ndas


def invalid_string(da):
    if isinstance(da, str):
        return False
    return True


def invalid_statement(da):
    """
    sd: Statement-non-opinion
    sv: Statement-opinion
    """
    # Prev is not a statement
    if re.search("s", da):
        return False
    return True


def first_sentence_contain_question(das):
    for da in das[:-1]:
        if "q" in da:
            return True
    return False


def invalid_question(da):
    # if re.search("q", da):
    #     return False
    if da in VALID_QS:
        return False
    return True


def invalid_silence(d, sample):
    duration = d["end"] - d["start"]
    vad_list = get_vad_list_subset(
        sample["vad_list"], start_time=d["start"], end_time=d["end"]
    )
    vad = vad_list_to_onehot(vad_list, duration, frame_hz=50)
    channel = 0 if d["speaker"] == "A" else 1
    vv = vad[:, channel]
    _, _, val = find_island_idx_len(vv)
    if len(val) <= 2:
        return True
    return False


def invalid_pause_prior_question(d):
    qidx = get_question_start_idx(d["da"])
    ps = d["ends"][qidx - 1]
    pe = d["starts"][qidx]
    if pe - ps > 0:
        return False
    return True


def other_is_active_old(d, sample):
    vad_list = get_vad_list_subset(
        sample["vad_list"], start_time=d["starts"][0], end_time=d["ends"][-1]
    )
    other_channel = 1 if d["speaker"] == "A" else 0
    if len(vad_list[other_channel]) > 0:
        return True
    return False


def other_is_active(d, sample, n_prev, n_last, pad=0.5):
    """
    Find the end-point of the first sentence
    Subtract 0.5 seconds to get a starting point before the pause
    Make sure there is only a single speaker after that point
    """
    t0 = d["starts"][0]
    try:
        t1 = d["starts"][n_prev - 1]
    except:
        for w, s, e in zip(d["words"], d["starts"], d["ends"]):
            print(w, s, e)
        print("n_prev: ", n_prev)
        print("n_last: ", n_last)
        print("starts: ", len(d["starts"]))
        print("words: ", len(d["words"]))
        print("da: ", len(d["da"]))
        input()

    start_time = t1 - pad
    if start_time < t0:
        start_time = t0

    vad_list = get_vad_list_subset(
        sample["vad_list"], start_time=start_time, end_time=d["ends"][-1]
    )

    other_channel = 1 if d["speaker"] == "A" else 0
    if len(vad_list[other_channel]) > 0:
        return True
    return False


# TODO:
def create_human_human_samples(
    df_path: str = "data/human_human_samples/human_baseline.pkl",
    silence_duration: float = 0.4,
    post_silence: float = 1.0,
    pre_silence: float = 0.2,
    sample_rate: int = 16000,
    root: str = "data/human_human_samples",
):
    """
    Extract human samples (audio/vap/txt)
    """

    # Create paths
    vap_root = join(root, "vap")
    audio_root = join(root, "audio")
    alignment_root = join(root, "alignment")  # not done here but placeholder
    Path(vap_root).mkdir(parents=True, exist_ok=True)
    Path(audio_root).mkdir(parents=True, exist_ok=True)

    # Load model and data
    model, device = load_model()
    reader = SwbReader()
    df = pd.read_pickle(df_path)

    # helpers
    def add_listener_channel(x):
        z = torch.zeros_like(x)
        return torch.stack((x, z))

    # Create silence audio samples
    pre_silence_samples = int(pre_silence * sample_rate)
    post_q_samples = int(post_silence * sample_rate)
    pause_silence_samples = int(silence_duration * sample_rate)

    pre = torch.zeros(pre_silence_samples)
    pause = torch.zeros(pause_silence_samples)
    post = torch.zeros(post_q_samples)

    # Iterate over all human-human samples (with pause):
    # Create a sample version where we have a pause of a fixed length,
    # masked audio (vad) and where the relevant speaker is always in channel 0
    data = []
    skipped_mismatch = 0
    for ii in tqdm(range(len(df))):

        d = df.iloc[ii]
        channel = 0 if d.speaker == "A" else 1
        words = d.words
        starts = torch.tensor(d.starts)
        ends = torch.tensor(d.ends)
        audio_path = reader.get_session(d.session)["audio_path"]
        qidx = get_question_start_idx(d["da"])

        if len(words) != len(d.da):
            skipped_mismatch += 1
            continue

        # Load speaker audio
        audio, _ = load_waveform(audio_path, start_time=starts[0], end_time=ends[-1])
        audio = audio[channel]

        # Extract times for the two parts
        ends = ends - starts[0]
        starts = starts - starts[0]

        # Extract separate timings for first/last sentence
        first_words = words[:qidx]
        first_starts = starts[:qidx]
        first_ends = ends[:qidx]
        last_words = words[qidx:]
        last_starts = starts[qidx:]
        last_ends = ends[qidx:]

        pause_start = first_ends[-1]
        pause_end = last_starts[0]

        last_rel_first = pause_end - pause_start

        pause_start_sample = int(sample_rate * pause_start)
        pause_end_sample = int(sample_rate * pause_end)

        # Create/extract the corresponding audio (from speaker channel)
        first_audio = audio[:pause_start_sample]
        last_audio = audio[pause_end_sample:]
        new_audio = torch.cat((pre, first_audio, pause, last_audio, post), dim=0)

        # Relative times for the new audio
        first_starts += pre_silence
        first_ends += pre_silence

        # TODO:
        last_starts += pre_silence + silence_duration - last_rel_first
        last_ends += pre_silence + silence_duration - last_rel_first

        # new_starts = torch.cat((first_starts, last_starts))
        # new_ends = torch.cat((first_ends, last_ends))

        # Add silent listener channel and add batch dimension
        new_audio = add_listener_channel(new_audio).unsqueeze(0)
        out = model.probs(new_audio.to(device))

        # Get new entries for data
        audio_path = join(audio_root, d.utt_idx + ".wav")
        txt_path = join(audio_root, d.utt_idx + ".txt")
        vap_path = join(vap_root, d.utt_idx + ".json")
        tg_path = join(alignment_root, d.utt_idx + ".TextGrid")

        new_d = d.to_dict()
        new_d["audio_path"] = audio_path
        new_d["vap_path"] = vap_path
        new_d["txt_path"] = txt_path
        new_d["tg_path"] = tg_path
        new_d["first_words"] = first_words
        new_d["last_words"] = last_words
        new_d.pop("starts")
        new_d.pop("ends")
        # new_d["starts"] = [round(t, 2) for t in new_starts.tolist()]
        # new_d["ends"] = [round(t, 2) for t in new_ends.tolist()]
        new_d["first_starts"] = [round(t, 2) for t in first_starts.tolist()]
        new_d["first_ends"] = [round(t, 2) for t in first_ends.tolist()]
        new_d["last_starts"] = [round(t, 2) for t in last_starts.tolist()]
        new_d["last_ends"] = [round(t, 2) for t in last_ends.tolist()]
        data.append(new_d)

        # Save audio and output
        text = " ".join(words)
        write_txt([text], txt_path)
        save_output_json(out, vap_path)
        torchaudio.save(audio_path, new_audio[0, :1], sample_rate=sample_rate)

    print("Skipped Mismatch: ", skipped_mismatch)
    new_df = pd.DataFrame(data)

    new_df_path = join(root, "human_samples.pkl")
    new_df.to_pickle(new_df_path)
    print("Saved human samples -> ", new_df_path)
    print("Don't forget to run alignment!")


def debug():
    import matplotlib.pyplot as plt
    import sounddevice as sd
    from vap.plot_utils import plot_vap
    from vap_tts.utils import load_audio, load_text_grid, load_vap_output
    from vap_tts.results import get_volume

    def fix_vad(vad, out):
        out_frames = out["p_now"].shape[1]
        vad_frames = vad.shape[0]
        if vad_frames != out_frames:
            if vad_frames > out_frames:
                vad = vad[:out_frames]
            else:
                diff = out_frames - vad_frames
                vad = torch.cat((vad, torch.zeros(diff, 2)), dim=0)
        return vad

    def find_human_pause(d, tg, pad=0.02):
        first_target = d.first_words[-1].lower()
        second_target = d.last_words[0].lower()
        min_second = len(d.first_words)
        start, end = -pad, pad
        for ii, (s, e, w) in enumerate(
            zip(tg["words"]["starts"], tg["words"]["ends"], tg["words"]["words"])
        ):
            if w == first_target:
                start = e
            elif ii > min_second and w == second_target:
                end = s
        return start + pad, end - pad

    def tg_to_vad_list(tg):
        ch_vad = []
        for s, e, w in zip(
            tg["words"]["starts"], tg["words"]["ends"], tg["words"]["words"]
        ):
            if w != "":
                ch_vad.append([s, e])
        return [ch_vad, []]

    def times_to_vad_list(d):
        v = [(s, e) for s, e in zip(d.first_starts, d.first_ends)]
        v += [(s, e) for s, e in zip(d.last_starts, d.last_ends)]
        return [v, []]

    def classify_silence(vn, vf):
        # now is positive -> weak or shift
        if vn > 0:
            if vf > 0:
                return "shift"
            else:
                return "weak_shift"
        return "hold"

    root = "data/human_human"
    df = pd.read_pickle("data/human_human/data_vap.pkl")

    fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
    total = 0
    stats = {"shift": 0, "weak_shift": 0, "hold": 0}
    fstats = {"shift": 0, "weak_shift": 0, "hold": 0}
    filler = 0
    for i in range(len(df)):
        d = df.iloc[i]
        audio = load_audio(join(root, d.audio_path), pad=0)
        out = load_vap_output(join(root, d.vap_path))
        ps = d.first_ends[-1] + 0.02
        pe = d.last_starts[0] - 0.02
        vnp = get_volume(out["p_now"], ps, pe)
        vfp = get_volume(out["p_future"], ps, pe)
        vns = get_volume(out["p_now"], d.last_ends[-1], d.last_ends[-1] + 0.4)
        vfs = get_volume(out["p_future"], d.last_ends[-1], d.last_ends[-1] + 0.4)
        cl = classify_silence(vnp, vfp)
        stats[cl] += 1
        scl = classify_silence(vns, vfs)
        fstats[scl] += 1
        total += 1
        fig, ax = plot_vap(
            waveform=audio[0],
            p_now=out["p_now"][0, :, 0],
            p_fut=out["p_future"][0, :, 0],
        )
        for a in ax:
            a.axvline(ps, color="red", linestyle="--")
            a.axvline(pe, color="red", linestyle="--")
            a.axvline(d.last_ends[-1], color="green", linestyle="--")
            a.axvline(d.last_ends[-1] + 0.4, color="green", linestyle="--")
        ax[0].set_title(f"Pause -> {cl}, Shift -> {scl}")
        print("Pause: ", cl)
        print("Shift: ", scl)
        print("-" * 20)
        plt.show()
        if d.first_words[-1].lower() in ["uh", "um"]:
            filler += 1
    v = torch.tensor(list(stats.values())) / total
    v = v.tolist()
    hs = round(v[-1] / v[0], 2)
    hw = round(v[-1] / v[1], 2)
    ws = round(v[1] / v[0], 2)
    ax.bar(stats.keys(), v, color="teal", alpha=0.5)
    ax.set_title(f"Total: {total}, h/s: {hs}, h/w: {hw}, ws: {ws}")
    print("h/s: ", hs)
    print("h/w: ", hw)
    print("w/s: ", ws)
    print("Filler: ", filler)
    print("Total: ", total)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()

    # plot a single example
    ii = 0
    d = df.iloc[ii]

    audio = load_audio(join(root, d.audio_path), pad=0)
    out = load_vap_output(join(root, d.vap_path))
    ps = d.first_ends[-1] + 0.02
    pe = d.last_starts[0] - 0.02
    vnp = get_volume(out["p_now"], ps, pe)
    vns = get_volume(out["p_now"], d.last_ends[-1], d.last_ends[-1] + 0.4)
    vfp = get_volume(out["p_future"], ps, pe)
    vfs = get_volume(out["p_future"], d.last_ends[-1], d.last_ends[-1] + 0.4)
    cl = classify_silence(vnp, vfp)
    scl = classify_silence(vns, vfs)

    sd.play(audio[0, 0], samplerate=16_000)


class HumanBaseline:
    def __init__(self, root: str = "data/human_human_samples"):
        self.root = root
        self.df_path = join(root, "human_samples.pkl")
        self.df_vap_path = join(root, "data_vap.pkl")
        self.reader = SwbReader()

        # Placeholders for VAP-model/device
        self.model = None
        self.device = None

        if exists(self.df_path):
            self.df = pd.read_pickle(self.df_path)
            print("Loaded existing samples")

    def extract_human_samples(
        self,
        first_is_statement: bool = True,
        pause: bool = True,
        activity: bool = True,
        min_first_words: int = 5,
        min_last_words: int = 5,
        save: bool = True,
    ):
        print("first_is_statement: ", first_is_statement)
        print("min_first_words: ", min_first_words)
        print("min_last_words: ", min_last_words)
        print("pause: ", pause)
        print("activity: ", activity)
        print("save: ", save)
        input("Continue? (Ctrl-c to abort)")

        questions = []
        error = {
            "length": 0,
            "statement": 0,
            "invalid_prev": 0,
            "question": 0,
            "question_in_first": 0,
            "first_n_words": 0,
            "last_n_words": 0,
            "silence": 0,
            "other_active": 0,
        }

        def no_statement_in_prev(das):
            ret = True
            for _da in das[:-1]:
                if _da.startswith("s"):
                    ret = False
            return ret

        def invalid_prev(prev):
            """
            aa: Agree/Accept -> "That's exactly it."
            bh: Backchannel in question form -> "Is that right?"
            bf: Summarize/reformulate -> "Oh, you mean you switched schools for the kids."
            ad: Action-directive -> "Why don't you go first"
            br: Signal-non-understanding -> "Excuse me?"
            fp: Conventional-opening -> "How are you?"
            qrr: Or-Clause -> "or is it more of a company?"
            ^g: Tag-Question ->"Right?"
            """
            for invalid in ["aa", "bh", "bf", "ad", "br", "fp", "qrr", "^g"]:
                if invalid in prev:
                    return True
            return False

        for session in tqdm(self.reader.test_sessions):
            sample = self.reader.get_session(session, include_anno=True)
            for d in sample["dialog"]:

                # 'nan' dialog-act is interpreted as float
                da = [str(f) for f in d["da"]]
                das, ndas = get_unique_da_ordered(da)

                # Q in das
                if len(das) < 2:
                    error["length"] += 1
                    continue

                # TODO: at least min words
                n_prev = sum([a for a in ndas[:-1]])
                n_last = ndas[-1]

                if n_prev < min_first_words:
                    error["first_n_words"] += 1
                    continue

                if n_last < min_last_words:
                    error["last_n_words"] += 1
                    continue

                last = das[-1]  # e.g. qy
                if invalid_question(last):
                    error["question"] += 1
                    continue

                prev = das[-2]  # e.g. sv
                if first_is_statement:
                    # if invalid_statement(prev):
                    if no_statement_in_prev(das):
                        error["statement"] += 1
                        continue

                if invalid_prev(prev):
                    error["invalid_prev"] += 1
                    continue

                if first_sentence_contain_question(das):
                    error["question_in_first"] += 1
                    continue

                # Activity
                if pause:
                    if invalid_pause_prior_question(d):
                        # if invalid_silence(d, sample):
                        error["silence"] += 1
                        continue

                if activity:
                    if other_is_active(d, sample, n_prev, n_last, pad=0.5):
                        error["other_active"] += 1
                        continue

                # Add valid example
                d["session"] = session
                d["sentences"] = [
                    [w for w in d["words"][:n_prev]],
                    [w for w in d["words"][n_prev:]],
                ]
                d["da_unique"] = das
                d["n_prev"] = n_prev
                d["n_last"] = n_last
                questions.append(d)

        for k, v in error.items():
            print(f"{k}: {v}")
        print("TOTAL FOUND: ", len(questions))
        questions.sort(key=lambda x: x["start"])
        self.df = pd.DataFrame(questions)

        if save:
            Path(self.root).mkdir(exist_ok=True, parents=True)
            self.df.to_pickle(self.df_path)
            print("Save human samples ALL -> ", self.df_path)
        return self.df

    def extract_human_audio_vap(
        self,
        df,
        silence_duration: float = 0.4,
        post_silence: float = 1.0,
        pre_silence: float = 0.2,
        sample_rate: int = 16000,
        save: bool = True,
    ):
        """
        Extract human samples (audio/vap/txt)
        """

        # Create paths
        vap_root = join(self.root, "vap")
        audio_root = join(self.root, "audio")
        alignment_root = join(self.root, "alignment")  # not done here but placeholder
        Path(vap_root).mkdir(parents=True, exist_ok=True)
        Path(audio_root).mkdir(parents=True, exist_ok=True)

        # Load model and data
        if self.model is None:
            self.model, self.device = load_model()

        # helpers
        def add_listener_channel(x):
            z = torch.zeros_like(x)
            return torch.stack((x, z))

        # Create silence audio samples
        pre_silence_samples = int(pre_silence * sample_rate)
        post_q_samples = int(post_silence * sample_rate)
        pause_silence_samples = int(silence_duration * sample_rate)

        pre = torch.zeros(pre_silence_samples)
        pause = torch.zeros(pause_silence_samples)
        post = torch.zeros(post_q_samples)

        # Iterate over all human-human samples (with pause):
        # Create a sample version where we have a pause of a fixed length,
        # masked audio (vad) and where the relevant speaker is always in channel 0
        data = []
        error = {"word_da_mismatch": 0}
        for ii in tqdm(range(len(df))):

            d = df.iloc[ii]
            channel = 0 if d.speaker == "A" else 1
            words = d.words
            starts, ends = torch.tensor(d.starts), torch.tensor(d.ends)
            audio_path = self.reader.get_session(d.session)["audio_path"]
            qidx = get_question_start_idx(d["da"])

            if len(words) != len(d.da):
                error["word_da_mismatch"] += 1
                continue

            # Load speaker audio
            audio, _ = load_waveform(
                audio_path, start_time=starts[0], end_time=ends[-1]
            )
            audio = audio[channel]

            # Extract times for the two parts
            ends = ends - starts[0]
            starts = starts - starts[0]

            # Extract separate timings for first/last sentence
            first_words = words[:qidx]
            first_starts = starts[:qidx]
            first_ends = ends[:qidx]

            last_words = words[qidx:]
            last_starts = starts[qidx:]
            last_ends = ends[qidx:]

            pause_start = first_ends[-1]
            pause_end = last_starts[0]

            last_rel_first = pause_end - pause_start

            pause_start_sample = int(sample_rate * pause_start)
            pause_end_sample = int(sample_rate * pause_end)

            # Create/extract the corresponding audio (from speaker channel)
            first_audio = audio[:pause_start_sample]
            last_audio = audio[pause_end_sample:]
            new_audio = torch.cat((pre, first_audio, pause, last_audio, post), dim=0)

            # Relative times for the new audio
            first_starts += pre_silence
            first_ends += pre_silence

            # TODO:
            last_starts += pre_silence + silence_duration - last_rel_first
            last_ends += pre_silence + silence_duration - last_rel_first

            # new_starts = torch.cat((first_starts, last_starts))
            # new_ends = torch.cat((first_ends, last_ends))

            # Add silent listener channel and add batch dimension
            new_audio = add_listener_channel(new_audio).unsqueeze(0)
            out = self.model.probs(new_audio.to(self.device))

            new_d = d.to_dict()
            new_d["audio_path"] = f"audio/{d.utt_idx}.wav"
            new_d["vap_path"] = f"vap/{d.utt_idx}.json"
            new_d["txt_path"] = f"audio/{d.utt_idx}.txt"
            new_d["tg_path"] = f"alignment/{d.utt_idx}.TextGrid"
            new_d["first_words"] = first_words
            new_d["last_words"] = last_words
            new_d.pop("starts")
            new_d.pop("ends")
            # new_d["starts"] = [round(t, 2) for t in new_starts.tolist()]
            # new_d["ends"] = [round(t, 2) for t in new_ends.tolist()]
            new_d["first_starts"] = [round(t, 2) for t in first_starts.tolist()]
            new_d["first_ends"] = [round(t, 2) for t in first_ends.tolist()]
            new_d["last_starts"] = [round(t, 2) for t in last_starts.tolist()]
            new_d["last_ends"] = [round(t, 2) for t in last_ends.tolist()]
            data.append(new_d)

            text = " ".join(words)

            # Save audio and output
            write_txt([text], join(audio_root, d.utt_idx + ".txt"))
            save_output_json(out, join(vap_root, d.utt_idx + ".json"))
            torchaudio.save(
                join(audio_root, d.utt_idx + ".wav"),
                new_audio[0, :1],
                sample_rate=sample_rate,
            )

        for k, v in error.items():
            print(f"{k}: {v}")

        new_df = pd.DataFrame(data)
        if save:
            new_df.to_pickle(self.df_vap_path)
            print("Saved human samples -> ", self.df_vap_path)
            print("Don't forget to run alignment!")
        return new_df

    def extract_results(self, df, min_start_time=0, plot_samples: bool = False):
        def classify_silence(vn, vf):
            # now is positive -> weak or shift
            if vn > 0:
                if vf > 0:
                    return "shift"
                else:
                    return "weak_shift"
            return "hold"

        stats = {"shift": 0, "weak_shift": 0, "hold": 0, "total": 0, "filler": 0}
        fstats = {"shift": 0, "weak_shift": 0, "hold": 0, "total": 0}
        for i in tqdm(range(len(df)), desc="results"):
            d = df.iloc[i]
            if d.start < min_start_time:
                continue
            audio = load_audio(join(self.root, d.audio_path), pad=0)
            out = load_vap_output(join(self.root, d.vap_path))
            ps = d.first_ends[-1] + 0.02
            pe = d.last_starts[0] - 0.02
            vnp = get_volume(out["p_now"], ps, pe)
            vfp = get_volume(out["p_future"], ps, pe)
            vns = get_volume(out["p_now"], d.last_ends[-1], d.last_ends[-1] + 0.4)
            vfs = get_volume(out["p_future"], d.last_ends[-1], d.last_ends[-1] + 0.4)
            cl = classify_silence(vnp, vfp)
            stats[cl] += 1
            stats["total"] += 1
            if d.first_words[-1].lower() in ["uh", "um"]:
                stats["filler"] += 1

            scl = classify_silence(vns, vfs)
            fstats[scl] += 1
            fstats["total"] += 1

            if plot_samples:
                print(d.sentences[0])
                print(d.sentences[1])
                print(d.da_unique)
                print(d.start)
                print("Pause: ", cl)
                print("Shift: ", scl)
                print("-" * 20)
                fig, ax = plot_vap(
                    waveform=audio[0],
                    p_now=out["p_now"][0, :, 0],
                    p_fut=out["p_future"][0, :, 0],
                    plot=False,
                )
                for a in ax:
                    a.axvline(ps, color="red", linestyle="--")
                    a.axvline(pe, color="red", linestyle="--")
                    a.axvline(d.last_ends[-1], color="green", linestyle="--")
                    a.axvline(d.last_ends[-1] + 0.4, color="green", linestyle="--")
                ax[0].set_title(f"Pause -> {cl}, Shift -> {scl}")
                plt.show()
        return stats, fstats

    @staticmethod
    def plot_result_hist(stats, plot=False):
        # v = torch.tensor(list(stats.values())) / stats["total"]
        v = torch.tensor([stats["shift"], stats["weak_shift"], stats["hold"]]).float()
        v /= stats["total"]
        v = v.tolist()
        fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
        hs = round(v[-1] / v[0], 2)
        hw = round(v[-1] / v[1], 2)
        ws = round(v[1] / v[0], 2)
        ax.bar(["Shift", "Weak-Shift", "Hold"], v, color="teal", alpha=0.5)
        ax.set_title(f"Total: {stats['total']}, h/s: {hs}, h/w: {hw}, ws: {ws}")
        print("h/s: ", hs)
        print("h/w: ", hw)
        print("w/s: ", ws)
        if "filler" in stats:
            print("Filler: ", stats["filler"])
        print("Total: ", stats["total"])
        ax.set_ylim([0, 1])
        plt.tight_layout()

        if plot:
            plt.pause(0.1)
        return fig, ax


# TODO: use argumentparser and lazy load model, relative paths in df
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="data/human_human")
    parser.add_argument("--n_prev", type=int, default=3)
    parser.add_argument("--n_last", type=int, default=3)
    parser.add_argument("--first_is_statement", action="store_true")
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--min_start_time", type=float, default=5)
    args = parser.parse_args()

    # Most examples are like a filler/acknowledgement then a question e.g.
    # A: I have 5 dogs
    # B: oh! what kinds of dogs do you have?
    # A: ....

    if args.results:

        print("Human results")
        hh = HumanBaseline(root=args.root)

        if not exists(hh.df_vap_path):
            ndf = hh.extract_human_audio_vap(hh.df)
        else:
            ndf = pd.read_pickle(hh.df_vap_path)

        print("#" * 40)
        print(ndf)
        print("#" * 40)

        pause_stats, shift_stats = hh.extract_results(
            ndf, min_start_time=args.min_start_time, plot_samples=False
        )
        fig, ax = hh.plot_result_hist(pause_stats)
        plt.show()

    else:

        hh = HumanBaseline(root=args.root)

        if not exists(hh.df_path):
            df = hh.extract_human_samples(
                first_is_statement=args.first_is_statement,
                min_first_words=1,
                min_last_words=1,
            )
        else:
            df = pd.read_pickle("data/human_human/human_samples.pkl")

        print("Found examples with n-prev and n-last lengths")
        for N in range(2, 5):
            for M in range(2, 5):
                ndf = df[df["n_prev"] >= N]
                ndf = ndf[ndf["n_last"] >= M]
                if N == args.n_prev and M == args.n_last:
                    print("#" * 40)
                    print(f"(prev, last) = {(N,M)} -> {len(ndf)}")
                    print("#" * 40)
                else:
                    print(f"(prev, last) = {(N,M)} -> {len(ndf)}")

        df = df[df["n_prev"] >= args.n_prev]
        df = df[df["n_last"] >= args.n_last]

        if not exists(hh.df_vap_path):
            ndf = hh.extract_human_audio_vap(df)
        else:
            ndf = pd.read_pickle(hh.df_vap_path)

        print("#" * 40)
        print(ndf)
        print("#" * 40)
        pause_stats, shift_stats = hh.extract_results(
            ndf, min_start_time=args.min_start_time, plot_samples=False
        )
        fig, ax = hh.plot_result_hist(pause_stats)
        plt.show()
