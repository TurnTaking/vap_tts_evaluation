import wave

import azure.cognitiveservices.speech as speechsdk
from vap.utils import read_json

"""
HowToSynthSpeech: https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/how-to-speech-synthesis?tabs=browserjs%2Cterminal&pivots=programming-language-python
SpeechSyntheisizer: https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.speechsynthesizer?view=azure-python
"""


VOICES = {
    "en-US-GuyNeural": [
        "newscast",
        "angry",
        "cheerful",
        "sad",
        "excited",
        "friendly",
        "terrified",
        "shouting",
        "unfriendly",
        "whispering",
        "hopeful",
    ],
    "en-US-JaneNeural": [
        "angry",
        "cheerful",
        "excited",
        "friendly",
        "hopeful",
        "sad",
        "shouting",
        "terrified",
        "unfriendly",
        "whispering",
    ],
    "en-US-JasonNeural": [
        "angry",
        "cheerful",
        "excited",
        "friendly",
        "hopeful",
        "sad",
        "shouting",
        "terrified",
        "unfriendly",
        "whispering",
    ],
    "en-US-NancyNeural": [
        "angry",
        "cheerful",
        "excited",
        "friendly",
        "hopeful",
        "sad",
        "shouting",
        "terrified",
        "unfriendly",
        "whispering",
    ],
    "en-US-SaraNeural": [
        "angry",
        "cheerful",
        "excited",
        "friendly",
        "hopeful",
        "sad",
        "shouting",
        "terrified",
        "unfriendly",
        "whispering",
    ],
    "en-US-TonyNeural": [
        "angry",
        "cheerful",
        "excited",
        "friendly",
        "hopeful",
        "sad",
        "shouting",
        "terrified",
        "unfriendly",
        "whispering",
    ],
}

VOICES_CHAT = {
    "en-US-AriaNeural": [
        "chat",
        "customerservice",
        "narration-professional",
        "newscast-casual",
        "newscast-formal",
        "cheerful",
        "empathetic",
        "angry",
        "sad",
        "excited",
        "friendly",
        "terrified",
        "shouting",
        "unfriendly",
        "whispering",
        "hopeful",
    ],
    "en-US-DavisNeural": [
        "chat",
        "angry",
        "cheerful",
        "excited",
        "friendly",
        "hopeful",
        "sad",
        "shouting",
        "terrified",
        "unfriendly",
        "whispering",
    ],
    "en-US-JennyNeural": [
        "assistant",
        "chat",
        "customerservice",
        "newscast",
        "angry",
        "cheerful",
        "sad",
        "excited",
        "friendly",
        "terrified",
        "shouting",
        "unfriendly",
        "whispering",
        "hopeful",
    ],
}


# Custom style only available through SSML
"""xml

<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
        xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
    <voice name="en-US-JennyNeural">
        <mstts:express-as style="chat">
            Hello, my name is Jenny and I am a chatbot. What's your name?
        </mstts:express-as>
    </voice>
</speak>

"""


class MicrosoftTTS:
    AUDIO_FORMATS = {
        8000: speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm,
        16000: speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
        24000: speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
        44100: speechsdk.SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm,
        48000: speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
    }
    CHANNELS = 1
    SAMPLE_WIDTH = 2

    def __init__(
        self,
        sample_rate: int = 16_000,
        normalize: bool = True,
        credentials: str = "credentials/MICROSOFT_SPEECH_CREDENTIALS.json",
    ):
        self.sample_rate = sample_rate
        self.normalize = normalize

        c = read_json(credentials)
        self.SPEECH_KEY = c["SPEECH_KEY"]
        self.SPEECH_REGION = c["SPEECH_REGION"]
        self.speech_config = speechsdk.SpeechConfig(
            subscription=c["SPEECH_KEY"], region=c["SPEECH_REGION"]
        )
        self.speech_config.speech_synthesis_language = "en-US"
        self.speech_config.set_speech_synthesis_output_format(
            self.AUDIO_FORMATS[self.sample_rate]
        )

    def __call__(
        self,
        text: str,
        filename: str,
        ssml: bool = False,
        voice_name: str = "en-US-JennyNeural",
    ) -> None:

        # audio_config=None -> as in-memory stream
        # WARNING: Must explicitly use "None" as argument
        self.speech_config.speech_synthesis_voice_name = voice_name
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None
        )

        if ssml:
            response = speech_synthesizer.speak_ssml(text)  # _async(text).get()
        else:
            response = speech_synthesizer.speak_text(text)  # _async(text).get()

        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(self.CHANNELS)  # nchannels
            wav_file.setsampwidth(self.SAMPLE_WIDTH)  # sampwidth
            wav_file.setframerate(self.sample_rate)  # framerate
            wav_file.writeframes(response.audio_data)


if __name__ == "__main__":

    import sounddevice as sd
    import matplotlib.pyplot as plt
    from vap.plot_utils import plot_vap
    from vap_tts.utils import load_model, load_audio

    tts = MicrosoftTTS()
    text = "yes, well I tend to agree with you"
    tts(text, filename="microsoft.wav")

    model, device = load_model()
    audio = load_audio("microsoft.wav")
    out = model.probs(audio.to(device))

    sd.play(audio[0, 0], samplerate=16_000)

    fig, _ = plot_vap(
        audio[0],
        p_now=out["p_now"][0, :, 0],
        p_fut=out["p_future"][0, :, 0],
        vad=out["vad"][0],
        plot=False,
    )
    fig.suptitle("ORIGINAL")
    plt.show()
