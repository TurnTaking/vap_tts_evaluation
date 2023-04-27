import xml.dom.minidom
from vap_tts.utils import extract_sentences


def prosody_ssml(text, volume="x-loud", rate="x-slow"):
    return f'<prosody volume="{volume}" rate="{rate}">{text}</prosody>'


def split_turn(text: str, n_prosody_words: int = 1):
    sents = extract_sentences(text)
    s1 = sents["text"][0].split(" ")
    a = " ".join(s1[:-n_prosody_words])
    b = " ".join(s1[-n_prosody_words:])
    # Lowercase first letter of next sentence
    c = sents["text"][1]
    c = c[0].lower() + c[1:] + "?"
    return a, b, c


def get_ssml_string(a, b, c, volume="x-loud", rate="x-slow", pause_time=100):
    ssml = "<speak>"
    ssml += a + " " + prosody_ssml(b, volume=volume, rate=rate)
    ssml += f'<break time="{pause_time}ms"/>' + " " + c
    ssml += "</speak>"
    return ssml


def xml_format_ssml(ssml):
    ssml_xml = xml.dom.minidom.parseString(ssml)
    return ssml_xml.toprettyxml()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_pickle("data/multiwoz_tts_utts_dev_100.pkl")
    text = df.iloc[0].text

    a, b, c = split_turn(text, n_prosody_words=2)
    ssml = get_ssml_string(a, b, c)
    ssml_xml = xml_format_ssml(ssml)
    print(text)
    print("A: ", a)
    print("B: ", b)
    print("C: ", c)
    print()
    print(ssml)
    print()
    print(ssml_xml)

    from vap_tts.tts_microsoft import MicrosoftTTS

    tts = MicrosoftTTS()
