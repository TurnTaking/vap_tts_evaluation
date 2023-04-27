import re
from typing import List

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from vap_tts.utils import extract_sentences
from vap_tts.syllables import number_of_syllables

"""
MultiWOZ 2.2: 
A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines
- Original paper: https://aclanthology.org/D18-1547.pdf
- https://aclanthology.org/2020.nlp4convai-1.13.pdf

"
The MultiDomain Wizard-of-Oz dataset (MultiWOZ), 
a fully-labeled collection of human-human written conversations 
spanning over multiple domains and topics. 
10k dialogs.
"

Stats from (original) paper
----------------
#Dialogs               8,438 
Total#turns            115,424 
Total#tokens           1,520,970
Avg.turns_per_dialogue 13.68
Avg.tokens_per_turn    13.18
Total_unique_tokens    24,071
#Slots                 25
#Values                4510

---------------------------------------

https://huggingface.co/datasets/multi_woz_v22/blob/main/README.md#dataset-description

They define


name: speaker
dtype:
  class_label:
    names:
      '0': USER  <---------- The user seems to be 0 when looking and they say it is so we go with that
      '1': SYSTEM

TODO
* concatenate consecutive turns from same speaker
* dialog acts
* FILTER
    * Look for multiple sentences by system
        * Look for multiple dialog acts from system
    * Omit sentences with numbers, dates, times
    * Omit by length
        - more than 50
        - less than 150?


sample_id: DialogID_TurnID_SpeakerID
sample_id, text, dialog_act, id, turn_id, speaker_id, role?, services
"""

REMOVE_COLUMNS = ["services", "dialogue_id", "turns"]

N_SENTENCES: int = 2
MIN_FIRST_SENT_WORDS: int = 5
MIN_LAST_SENT_WORDS: int = 5
MIN_TEXT_LEN: int = 50
MAX_TEXT_LEN: int = 250


def calc_n_sentences(text_list: List[str]) -> List[int]:
    """
    Count 'sentences' by counting periods (.!?)

    1. Condense duplicate punctuations  i.e. !! -> !, ?? -> ?, ... -> .
    2. Count how many times they ocur
    3. Minimum sentences is 1 (even if they omitted punctuation).
        Previous steps ensures 'No empty turns'
    """
    n_sentences = []
    for text in text_list:
        t = re.sub(r"([!?.]){2,}", r"\1", text)
        n = len(re.findall(r"([!?.])", t))
        n_sentences.append(max(1, n))
    return n_sentences


def encode(sample):
    """
    sample:
    dict_keys(['dialogue_id', 'services', 'turns'])
        dialogue_id: 'PMUL0698.json'
        services: ['restaurant', 'train']

        turns: : dict_keys(['turn_id', 'speaker', 'utterance', 'frames', 'dialogue_acts'])
            turn_id: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
            speaker: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            frames: [{service: ['restaurant'], 'state: .., 'slots':...}, ...]
            dialogue_acts: [{dialog_act: {'act_type': ['Restaurant-Recommend', 'general-reqmore']]}}]
            utterance: ['hello help me', 'sure, what can I do', ...]
    """
    all_text = sample["turns"]["utterance"]
    all_speaker = sample["turns"]["speaker"]
    all_da = [d["dialog_act"]["act_type"] for d in sample["turns"]["dialogue_acts"]]
    all_turn_id = sample["turns"]["turn_id"]

    # Concatenate consecutive turns from same speaker
    # WARNING: There does not seem to be any in this dataset but kept here for sanity
    text, speaker, da, tid = (
        [all_text[0]],
        [all_speaker[0]],
        [all_da[0]],
        [all_turn_id[0]],
    )
    for t, sp, da_, ti in zip(
        all_text[1:], all_speaker[1:], all_da[1:], all_turn_id[1:]
    ):
        if sp == speaker[-1]:
            text[-1] = (text[-1] + " " + t).strip()
            da[-1] += da_
            tid[-1] += [ti]
        else:
            text.append(t)
            speaker.append(sp)
            da.append(da_)
            tid.append(ti)

    n_sentences = calc_n_sentences(text)
    speaker = ["USER" if s == 0 else "SYSTEM" for s in speaker]
    return {
        "text": text,
        "speaker": speaker,
        "da": da,
        "id": sample["dialogue_id"],
        "turn_id": tid,
        "n_sentences": n_sentences,
    }


def load_multi_woz_df():
    dsets = load_dataset("multi_woz_v22").map(encode, remove_columns=REMOVE_COLUMNS)
    dset = concatenate_datasets([d for n, d in dsets.items()])
    return dset.to_pandas()


def save_df(df, path: str = "data/multiwoz.pkl"):
    """
    Pickle the objects directly and don't save as csv.

    Avoids loading the 'text' field (list of strings) as a single string (as with .csv files)
    https://stackoverflow.com/questions/23111990/pandas-dataframe-stored-list-as-string-how-to-convert-back-to-list
    """
    df.to_pickle(path)
    print("Save file -> ", path)


def load_df(path: str = "data/multiwoz.pkl"):
    return pd.read_pickle(path)


def create_pd(path: str):
    data = []
    for split in ["train", "validation", "test"]:
        dset = load_multi_woz(split)
        for d in tqdm(dset, desc=split):
            for t, n, s, da, tid in zip(
                d["text"], d["n_sentences"], d["speaker"], d["da"], d["turn_id"]
            ):
                if s == 0:
                    continue
                if n > 1:
                    data.append({"text": t, "da": da, "id": d["id"], "turn_id": tid})
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print("Saved data -> ", path)
    return df


# Conditions for using it in TTS
def invalid_number_of_sentences(ns: int, n_sentences: int = N_SENTENCES):
    if ns == n_sentences:
        return False
    return True


def contains_digits(text):
    if len(re.findall(r"\d", text)) > 0:
        return True
    return False


def is_user(speaker):
    if speaker.startswith("USER"):
        return True
    return False


def invalid_length(
    text, min_text_len: int = MIN_TEXT_LEN, max_text_len: int = MAX_TEXT_LEN
):
    if min_text_len <= len(text) <= max_text_len:
        return False
    return True


def invalid_first_sentence_words(
    sent_n_words, min_first_sent_words: int = MIN_FIRST_SENT_WORDS
):
    if sent_n_words[0] >= min_first_sent_words:
        return False
    return True


def invalid_last_sentence_words(
    sent_n_words, min_last_sent_words: int = MIN_LAST_SENT_WORDS
):
    if sent_n_words[-1] >= min_last_sent_words:
        return False
    return True


def invalid_number_of_syllables(word):
    # remove all punctations (,.?!)
    w = re.sub(r"[,.?!]", "", word)
    if number_of_syllables(w) == 1:
        return False
    return True


def includes_commas(sentence):
    if re.search(r",", sentence):
        return True
    return False


def includes_other_punctuation(sentence):
    if re.search(r"\s-\s", sentence):
        return True
    return False


def extract_valid_tts_turns(df):
    """
    Filter out turns that are not suitable for TTS training.

    Name: ID_TurnID_Speaker
    """
    data = []

    omit = {
        "n_sentences": 0,
        "digits": 0,
        "user": 0,
        "length": 0,
        "first_sent_words": 0,
        "last_sent_words": 0,
        "first_sent_commas": 0,
        "last_sent_commas": 0,
        "first_sent_other_puncts": 0,
        "last_sent_other_puncts": 0,
        "first_sent_syllables": 0,
        "last_sent_syllables": 0,
    }
    total = 0
    for dii in tqdm(range(len(df)), desc="Extract valid turns"):
        d = df.iloc[dii]
        for i in range(1, len(d.text)):
            text = d.text[i]
            ns = d.n_sentences[i]
            speaker = d.speaker[i]
            total += 1

            if is_user(speaker):
                omit["user"] += 1
                continue

            if invalid_number_of_sentences(ns):
                omit["n_sentences"] += 1
                continue

            if contains_digits(text):
                omit["digits"] += 1
                continue

            if invalid_length(text):
                omit["length"] += 1
                continue

            sentences = extract_sentences(text)

            if invalid_first_sentence_words(sentences["n"]):
                omit["first_sent_words"] += 1
                continue

            if invalid_last_sentence_words(sentences["n"]):
                omit["last_sent_words"] += 1
                continue

            if includes_commas(sentences["text"][0]):
                omit["first_sent_commas"] += 1
                continue

            if includes_commas(sentences["text"][1]):
                omit["last_sent_commas"] += 1
                continue

            if includes_other_punctuation(sentences["text"][0]):
                omit["first_sent_other_puncts"] += 1
                continue

            if includes_other_punctuation(sentences["text"][1]):
                omit["last_sent_other_puncts"] += 1
                continue

            last_word_first_sentence = sentences["text"][0].split()[-1]
            if invalid_number_of_syllables(last_word_first_sentence):
                omit["first_sent_syllables"] += 1
                continue

            last_word_last_sentence = sentences["text"][1].split()[-1]
            if invalid_number_of_syllables(last_word_last_sentence):
                omit["last_sent_syllables"] += 1
                continue

            das = d.da[i]
            prev_text = d.text[i - 1]
            tid = d.turn_id[i]
            sample_id = f"{d.id.replace('.json', '')}_{tid}_{speaker}"
            data.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "sentences": sentences["text"],
                    "punctuation": sentences["punctuation"],
                    "n_words": sentences["n"],
                    "dialog_act": das,
                    "turn_idx": i,
                    "turn_total": len(d.text),
                    "id": d.id,
                    "turn_id": tid,
                    "speaker_id": speaker,
                    "prev_text": prev_text,
                }
            )

    print("VALID TURNS")
    print(f"N: {len(df)} / {total}")
    print("-----------")
    print("Omitted:")
    for k, v in omit.items():
        print(f"{k}: {v}")
    print("-----------")
    return pd.DataFrame(data)


def main():
    df = load_multi_woz_df()

    # Extract TTS turns
    # Create dataset used for TTS generation
    ndf = extract_valid_tts_turns(df)

    # Only keep STATEMENT -> QUESTION pairs
    ss = ndf[ndf["punctuation"].isin([[".", "."]])]
    sq = ndf[ndf["punctuation"].isin([[".", "?"]])]
    qq = ndf[ndf["punctuation"].isin([["?", "?"]])]
    eq = ndf[ndf["punctuation"].isin([["!", "?"]])]
    print("STATEMENT -> STATEMENT: ", len(ss))
    print("QUESTION -> QUESTION: ", len(qq))
    print("EXCLAMATION -> QUESTION: ", len(eq))
    print("STATEMENT -> QUESTION: ", len(sq))

    save_df(sq, "data/multiwoz_tts_utts.pkl")

    # Create dev set
    dev_df = sq.iloc[:100]
    save_df(dev_df, "data/multiwoz_tts_utts_dev_100.pkl")


if __name__ == "__main__":
    main()
