# Amazon

**Default**
```python
voice_id: str = "Joanna",
engine: str = "neural",
language_code: str = "en-US",
```

* [NeuralTTS](https://docs.aws.amazon.com/polly/latest/dg/NTTS-main.html)
* [Speaking styles](https://docs.aws.amazon.com/polly/latest/dg/ntts-speakingstyles.html) 
    - Amazon polly neural tts have two speaking styles
        - Normal neural tts
        - NTTS Newscaster speaking style

# Microsoft

**Default**
```python
voice_name: str = "en-US-JennyNeural",
```

* [Voice styles and roles](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support?tabs=tts#voice-styles-and-roles)
    * "assistant"
    * "chat"
    * "customerservice"
    * "newscast"
    * "angry"
    * "cheerful"
    * "sad"
    * "excited"
    * "friendly"
    * "terrified"
    * "shouting"
    * "unfriendly"
    * "whispering"
    * "hopeful"
* Must use SSML to produce a voice with a 'style'
* 'chat' would be the best style? Friendly?


```html
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
        xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
    <voice name="en-US-JennyNeural">
        <mstts:express-as style="chat">
            Hello, my name is Jenny and I am a chatbot. What's your name?
        </mstts:express-as>
    </voice>
</speak>
```

# Google

```python
name="en-US-Neural2-C"
```

* [List of voices](https://cloud.google.com/text-to-speech/docs/voices)
* Can't control speaker style
    - can select `en-US-Neural2-[A-J]` which are different male/female voices
    - or `en-US-News-[K-n]` which have a style but it's not conversational
