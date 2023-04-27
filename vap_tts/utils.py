from typing import Tuple, Union

from textgrids import TextGrid
import torch
from torch import Tensor
from vap.audio import load_waveform
from vap.model import VapConfig, VapGPT
from vap.utils import read_json, write_json
import re

from vap.utils import everything_deterministic


everything_deterministic()


TG = dict[str, dict[str, Union[list[str], list[float]]]]
PUNCT_REGEX = r"[!?\.]"


def extract_sentences(text: str, punctuation: str = PUNCT_REGEX):
    """
    Finds the sentences in a "turn".
    1. operates on none consecutive punctuation
    2. iterates over all punctuation positions and extract sentences + punctuation
    """
    # omit consecutive punctuation
    text_no_duplicate = re.sub(r"([!?.]){2,}", r"\1", text)
    sentences = {"text": [], "punctuation": [], "n": []}
    last_start = 0
    for match in re.finditer(punctuation, text_no_duplicate):
        p_start = match.start()
        sent = text_no_duplicate[last_start:p_start].strip()
        sentences["text"].append(sent)
        sentences["punctuation"].append(match.group())
        sentences["n"].append(len(sent.split(" ")))
        last_start = p_start + 1
    return sentences


def load_text_grid(path: str) -> TG:
    grid = TextGrid(path)
    data = {
        "words": {"starts": [], "ends": [], "words": []},
        "phones": {"starts": [], "ends": [], "phones": []},
    }
    for word_phones, vals in grid.items():
        for loc, w in enumerate(vals):
            # omit first silence
            if loc == 0 and w.text == "":
                continue
            # omit last silence
            if loc == (len(vals) - 1) and w.text == "":
                continue
            # if w.text == "":
            #     continue
            data[word_phones]["starts"].append(w.xmin)
            data[word_phones]["ends"].append(w.xmax)
            data[word_phones][word_phones].append(w.text)
    return data


def load_audio(path: str, pad: float = 0, sample_rate: int = 16_000) -> Tensor:
    waveform, _ = load_waveform(path, sample_rate=sample_rate)
    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, torch.zeros_like(waveform)))

    if pad > 0:
        psamples = round(pad * sample_rate)
        waveform = torch.cat((waveform, torch.zeros(2, psamples)), dim=-1)
    return waveform.unsqueeze(0)


def load_vap_output(path: str) -> dict[str, Tensor]:
    out_list = read_json(path)
    out = {}
    for k, v in out_list.items():
        out[k] = torch.tensor(v)
    return out


def load_model(
    checkpoint: str = "data/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
) -> Tuple[VapGPT, str]:
    conf = VapConfig()
    model = VapGPT(conf)
    std = torch.load(checkpoint)
    model.load_state_dict(std, strict=False)
    model.eval()
    device = "cpu"
    if torch.cuda.is_available():
        model = model.to("cuda")
        device = "cuda"
    return model, device


def load_all_data_from_audio_path(
    audio_path: str, post_silence: float = 2.0
) -> Tuple[Tensor, TG, dict[str, Tensor]]:
    audio = load_audio(audio_path, pad=post_silence)
    tg = load_text_grid(
        audio_path.replace("/audio/", "/alignment/").replace(".wav", ".TextGrid")
    )
    out = load_vap_output(
        audio_path.replace("/audio/", "/vap/").replace(".wav", ".json")
    )
    return audio, tg, out


def save_output_json(out: dict[str, Tensor], path: str) -> None:
    data = {}
    for k, v in out.items():
        data[k] = v.cpu().tolist()
    write_json(data, path)
