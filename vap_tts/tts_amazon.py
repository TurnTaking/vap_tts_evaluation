from boto3 import Session
import wave

"""
CREDENTIALS: located in $HOME/.aws/credentials

https://docs.aws.amazon.com/polly/latest/dg/get-started-what-next.html

Encoding

AUDIO OUTPUT FORMAT: 16-bit signed 1 channel (mono) little-endian samples (Linear PCM)


Conversational Speaker style

'
New feature:
Amazon Polly makes the conversational speaking style the default version for the neural Matthew and Joanna voices. 
We removed references to the conversational speaking style.
' - June 28, 2021

"""

NORM_FACTOR = 2**15
VOICE_ID = [
    "Aditi",
    "Amy",
    "Astrid",
    "Bianca",
    "Brian",
    "Camila",
    "Carla",
    "Carmen",
    "Celine",
    "Chantal",
    "Conchita",
    "Cristiano",
    "Dora",
    "Emma",
    "Enrique",
    "Ewa",
    "Filiz",
    "Gabrielle",
    "Geraint",
    "Giorgio",
    "Gwyneth",
    "Hans",
    "Ines",
    "Ivy",
    "Jacek",
    "Ja n",
    "Joanna",
    "Joey",
    "Justin",
    "Karl",
    "Kendra",
    "Kevin",
    "Kimberly",
    "Lea",
    "Liv",
    "Lotte",
    "Lucia",
    "Lupe",
    "Mads",
    "Maja",
    "Marlene",
    "Mathieu",
    "Matthew",
    "Maxim",
    "Mia",
    "Miguel",
    "Mizuki",
    "Naja",
    "Nicole",
    "Olivia",
    "Penelope",
    "Raveena",
    "Ricardo",
    "Ru ben",
    "Russell",
    "Salli",
    "Seoyeon",
    "Takumi",
    "Tatyana",
    "Vicki",
    "Vitoria",
    "Zeina",
    "Zhiyu",
    "Aria",
    "Ayanda",
    "Arlet",
    "Hannah",
    "Arthur",
    "Daniel",
    "Liam",
    "Pedro",
    "Kajal",
    "Hiujin",
    "Laura",
    "Elin",
    "Ida",
    "Suvi",
    "Ola",
    "Hala",
]
LANG = [
    "arb",
    "cmn-CN",
    "cy-GB",
    "da-DK",
    "de-DE",
    "en-AU",
    "en-GB",
    "en-GB-WLS",
    "en-IN",
    "en-US",
    "es-ES",
    "es-MX",
    "es-US",
    "fr-CA",
    "fr-FR",
    "is-IS",
    "it-IT",
    "ja-JP",
    "hi-IN",
    "ko-KR",
    "nb-NO",
    "nl-NL",
    "pl-PL",
    "pt-BR",
    "pt-PT",
    "ro-RO",
    "ru -RU",
    "sv-SE",
    "tr-TR",
    "en-NZ",
    "en-ZA",
    "ca-ES",
    "de-AT",
    "yue-CN",
    "ar-AE",
    "fi-FI",
]


class AmazonTTS:
    ENCODING = "pcm"
    SAMPLE_WIDTH = 2
    CHANNELS = 1

    def __init__(self, sample_rate=16_000, normalize=True, profile_name="default"):
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.session = Session(profile_name=profile_name)
        self.client = self.session.client(
            service_name="polly",
            region_name=None,
            api_version=None,
            use_ssl=True,
            verify=None,
            endpoint_url=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            config=None,
        )

    def __call__(
        self,
        text: str,
        filename: str,
        ssml: bool = False,
        voice_id: str = "Joanna",
        engine: str = "neural",
        language_code: str = "en-US",
    ) -> None:
        """
        pcm -> wav:
            https://jun711.github.io/aws/convert-aws-polly-synthesized-speech-from-pcm-to-wav-format/
        """
        response = self.client.synthesize_speech(
            Text=text,
            OutputFormat=self.ENCODING,
            VoiceId=voice_id,
            SampleRate=str(self.sample_rate),
            TextType="ssml" if ssml else "text",
            LanguageCode=language_code,
            Engine=engine,
        )

        if "AudioStream" in response:
            with wave.open(filename, "wb") as wav_file:
                wav_file.setnchannels(self.CHANNELS)  # nchannels
                wav_file.setsampwidth(self.SAMPLE_WIDTH)  # sampwidth
                wav_file.setframerate(self.sample_rate)  # framerate
                wav_file.writeframes(response["AudioStream"].read())


if __name__ == "__main__":
    import sounddevice as sd
    import matplotlib.pyplot as plt
    from vap.plot_utils import plot_vap
    from vap_tts.utils import load_model, load_audio

    tts = AmazonTTS()
    text = "yes, well I tend to agree with you"
    tts(text, filename="amazon.wav")

    model, device = load_model()
    audio = load_audio("amazon.wav")
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
