from os import environ
from os.path import exists
from typing import List, Optional, Union

from google.cloud import texttospeech
from vap.audio import load_waveform

"""
SSML: https://cloud.google.com/text-to-speech/docs/ssml
AUDIO_CONFIG: https://cloud.google.com/text-to-speech/docs/reference/rest/v1beta1/text/synthesize#AudioConfig


AUDIO OUTPUT FORMAT: 16-bit signed 1 channel (mono) little-endian samples (Linear PCM)

WARNING: response.audio_content includes WAV-header so can't get bytes directly (for now)

# Only post-processing
----------------------
# wearable-class-device	Smart watches and other wearables, like Apple Watch, Wear OS watch
# handset-class-device	Smartphones, like Google Pixel, Samsung Galaxy, Apple iPhone
# headphone-class-device	Earbuds or headphones for audio playback, like Sennheiser headphones
# small-bluetooth-speaker-class-device	Small home speakers, like Google Home Mini
# medium-bluetooth-speaker-class-device	Smart home speakers, like Google Home
# large-home-entertainment-class-device	Home entertainment systems or smart TVs, like Google Home Max, LG TV
# large-automotive-class-device	Car speakers
# telephony-class-application	Interactive Voice Response (IVR) systems
"""


NORM_FACTOR = 2**15
# POSTPROCESSING
EFFECTS_PROFILE_ID = [
    "wearable-class-device",
    "handset-class-device",
    "headphone-class-device",
    "small-bluetooth-speaker-class-device",
    "medium-bluetooth-speaker-class-device",
    "large-home-entertainment-class-device",
    "large-automotive-class-device",
    "telephony-class-application",
]
PITCH = [-20, 20]
SPEAKER_RATE = [0.25, 4]
VOICE_TYPE = ["neural2", "wavenet", "news", "standard"]
NAMES = {
    "neural2": {
        "female": [
            "en-US-Neural2-C",
            "en-US-Neural2-E",
            "en-US-Neural2-F",
            "en-US-Neural2-G",
            "en-US-Neural2-H",
        ],
        "male": [
            "en-US-Neural2-A",
            "en-US-Neural2-D",
            "en-US-Neural2-I",
            "en-US-Neural2-J",
        ],
    },
    "wavenet": {
        "female": [
            "en-US-Wavenet-C",
            "en-US-Wavenet-E",
            "en-US-Wavenet-F",
            "en-US-Wavenet-G",
            "en-US-Wavenet-H",
        ],
        "male": [
            "en-US-Wavenet-A",
            "en-US-Wavenet-B",
            "en-US-Wavenet-D",
            "en-US-Wavenet-I",
            "en-US-Wavenet-J",
        ],
    },
    "standard": {
        "female": [
            "en-US-Standard-C",
            "en-US-Standard-E",
            "en-US-Standard-F",
            "en-US-Standard-G",
            "en-US-Standard-H",
        ],
        "male": [
            "en-US-Standard-A",
            "en-US-Standard-B",
            "en-US-Standard-D",
            "en-US-Standard-I",
            "en-US-Standard-J",
        ],
    },
}


class GoogleTTS:
    def __init__(
        self,
        sample_rate: int = 16_000,
        normalize: bool = True,
        credentials: str = "credentials/GOOGLE_SPEECH_CREDENTIALS.json",
    ) -> None:
        self._set_credentials(credentials)
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.client = texttospeech.TextToSpeechClient()
        self.tmp_path = "/tmp/google_tts.wav"

    def _set_credentials(self, credentials):
        assert exists(credentials), f"Credentials not found: {credentials}"

        environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

    def list_voices(self):
        return self.client.list_voices()

    def get_voice_subset(
        self,
        voice_type: str = "neural2",
        language_code: str = "en-US",
        gender: str = "female",
    ):
        assert (
            voice_type.lower() in VOICE_TYPE
        ), f"{voice_type} Not Founde. Use: {VOICE_TYPE}"
        voice_subset = []
        for voice in self.list_voices().voices:
            if voice.language_codes[0] != language_code:
                continue
            if voice_type not in voice.name.lower():
                continue
            if voice.ssml_gender.name.lower() != gender:
                continue

            if voice not in voice_subset:
                voice_subset.append(voice)

        voice_subset.sort(key=lambda x: x.name)
        return voice_subset

    def get_voice(
        self,
        voice_type: str = "neural2",
        language_code: str = "en-US",
        gender: str = "female",
    ):
        pass

    def get_audio_config(
        self,
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        effects_profile_id: Optional[Union[str, List[str]]] = None,
    ):
        """
        https://cloud.google.com/text-to-speech/docs/reference/rest/v1beta1/text/synthesize#AudioConfig
        """
        assert (
            SPEAKER_RATE[0] <= speaking_rate <= SPEAKER_RATE[1]
        ), f"speaking_rate must be in [0.25, 4.0] got {speaking_rate}"
        assert (
            PITCH[0] <= pitch <= PITCH[1]
        ), f"pitch must be in [-20, 20] (semitones) got {pitch}"

        if isinstance(effects_profile_id, str):
            effects_profile_id = [effects_profile_id]

        if effects_profile_id is not None:
            for eff_id in effects_profile_id:
                assert (
                    eff_id in EFFECTS_PROFILE_ID
                ), f"effects_profile_id: got {effects_profile_id} must be in {EFFECTS_PROFILE_ID}"

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
            sample_rate_hertz=self.sample_rate,
            effects_profile_id=effects_profile_id,
        )
        return audio_config

    def get_ssml_gender(self, gender):
        assert gender.lower() in [
            "male",
            "female",
        ], "Must provide gender: ['male', 'female']"

        ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
        if gender == "male":
            ssml_gender = texttospeech.SsmlVoiceGender.MALE
        return ssml_gender

    def response_to_torch(self, response):
        """
        Includes a WAV-header by default so this method is not correct
        def response_to_numpy(self, response):
            audio = np.frombuffer(response.audio_content, dtype=np.int16)
            if self.normalize:
                audio = audio * NORM_FACTOR
            return audio
        """
        with open(self.tmp_path, "wb") as out:
            out.write(response.audio_content)
        audio, _ = load_waveform(self.tmp_path, sample_rate=self.sample_rate)
        return audio[0]

    def __call__(
        self,
        text: str,
        filename: str,
        ssml: bool = False,
        gender: str = "female",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        effects_profile_id: Optional[Union[str, List[str]]] = None,
        language_code: str = "en-US",
        name="en-US-Neural2-C",
    ) -> None:
        assert (text is not None) or (
            ssml is not None
        ), "Requires either `text` or `ssml`"
        ssml_gender = self.get_ssml_gender(gender)
        audio_config = self.get_audio_config(speaking_rate, pitch, effects_profile_id)

        if ssml:
            input_text = texttospeech.SynthesisInput(ssml=text)
        else:
            input_text = texttospeech.SynthesisInput(text=text)

        # Note: the voice can also be specified by name.
        # Names of voices can be retrieved with client.list_voices().
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, name=name, ssml_gender=ssml_gender
        )

        response = self.client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        # The response's audio_content is binary.
        with open(filename, "wb") as out:
            out.write(response.audio_content)


if __name__ == "__main__":

    import sounddevice as sd
    import matplotlib.pyplot as plt
    from vap.plot_utils import plot_vap
    from vap_tts.utils import load_model, load_audio

    tts = GoogleTTS()
    text = "yes, well I tend to agree with you"
    tts(text, filename="google.wav")

    model, device = load_model()
    audio = load_audio("google.wav")
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
