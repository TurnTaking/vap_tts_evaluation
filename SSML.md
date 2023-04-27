# SSML

* [w3](https://www.w3.org/TR/2004/REC-speech-synthesis-20040907)
* [AmazonTTS](https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html)
* [MicrosoftTTS](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/speech-synthesis-markup-structure)
* [GoogleTTS](https://developers.google.com/assistant/conversational/ssml-beta)

### W3

#### [Prosody](https://www.w3.org/TR/2004/REC-speech-synthesis-20040907/#S3.2.4)
- Volume
    - Relative: "silent", "x-soft", "soft", "medium", "loud", "x-loud", or "default"
- pitch
    - x-low`, `low`, `medium`, `high`, `x-high`, or `default`
    - Relative changes can be given in 
        - Semitones: `+0.5st`, `+5st`, `-2st`
        - Hertz `+10Hz`, `-5.5Hz`
- Rate
    - `x-slow`, `slow`, `medium`, `fast`, `x-fast`, or `default`
    - `-10%`,`+10%` 

### [Amazon](https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html#prosody-tag)

**Prosody**
- Volume
    - W3 Relative: "silent", "x-soft", "soft", "medium", "loud", "x-loud", or "default"
    - Rel. Decibel: `+6db`, `-6db`
- Rate
    - Relative: `x-slow`, `slow`, `medium`, `fast`, `x-fast`, 
    - Percentage: `n%` -> 50%: half default, 100%: default, 200%: twice default 
- Pitch
    - WARNING: "Neural voices support the volume and rate attributes, but don't support the pitch attribute."
    - Relative: `x-low`, `low`, `medium`, `high`, `x-high`
    - Percentage: `n%` where 20 <= n <= 200  and 100% = default

### [Google](https://developers.google.com/assistant/conversational/ssml#prosody)

**Prosody**
- Volume
    - W3 Relative: "silent", "x-soft", "soft", "medium", "loud", "x-loud", or "default"
- Rate
    - W3 specification i.e.
    - Relative: `x-slow`, `slow`, `medium`, `fast`, `x-fast`, `default`
    - Percentage: `-10%`,`+10%` 
- Pitch
    - Relative: "low", "medium", "high"
    - Semitones: "-5st", "5st"
    - Percentage: "-5%", "+5%"

### [Microsoft](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/speech-synthesis-markup-voice#adjust-prosody)

**Prosody**
- Volume
    - W3 Relative: "silent", "x-soft", "soft", "medium", "loud", "x-loud", or "default"
    - number [0, 100], relative number e.g. `-10`, `+10`, percent "-15%"
    - `<prosody volume="+50%">`
    - `<prosody volume="-50%">`
- Rate
    - Relative number: 0.5, 1.0, 2.0 -> half, default, twice
    - Percentage: `+50%`, `-50%`
    - Value:  `x-slow`, `slow`, `medium`, `fast`, `x-fast`, `default`
- Pitch 
    - Hz: 
        - `<prosody pitch="500Hz">`
    - Relative: `x-low`, `low`, `medium`, `high`, `x-high`, `default`
        - `<prosody pitch="low">`
    - Percent: `-5%`, `+5%`
        - `<prosody pitch="+50%">`

