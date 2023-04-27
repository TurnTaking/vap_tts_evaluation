# vap_tts
TTS experiments using Voice Activity Projection


## Experiment
1. Extract written text from MultiWoz
    - Multiple phrases/sentences
    - Longer/Shorter
    - Filter: Numbers? Addresses? Dates?
2. Generate audio
    * MicrosoftTTS, AmazonTTS, GoogleTTS ->  "Neural" or BEST
    * Generate TTS samples and extract VAP output 
    * Align the text using `Montreal Forced Aligner`
3. Forced Alignment
    - extract 'exact' word timings from generated audio
4. Evaluate turn-taking performance
    - Turn-shift prob (avg, min, max) in silence / after last sentence
    - Propose "practical" fixes
        - Insert fillers
        - Insert <ssml> tags
        - Turn-hold/yield prosody
        - Evaluate 4.) again
5. Plot results
6. "Conversational" TTS systems sound more natural but lacks one of the key essence of 
    conversational dialogue: turn-taking


## Run

1. Extract text data: 
    - `python extract_text.py`
2. Generate TTS audio:
    - `python vap_tts/generate_audio.py`
    - `python vap_tts/apply_postprocessing.py`
3. Forced Alignment:
    - Original: `bash   scripts/forced_aligner.py data/GEN_DATA_ROOT/`
4. Apply silence normalization
    - `python vap_tts/apply_silence_norm.py`
5. Forced Alignment:
    - Silence:  `bash   scripts/forced_aligner.py data/GEN_DATA_ROOT/ silence`
6. Apply Hold-postprocessing:
    - `python vap_tts/apply_postprocessing.py`
7. Extract VAP output:
    - `python vap_tts/extract_vap.py`
8. Calculate results and plots:
    - `python results.py`


## Montreal Forced Alignment

1. Install: Create a new conda environment, source environment, install mfa and run:
    ```bash
    conda create -n aligner python=3
    conda activate aligner
    conda install -c conda-forge montreal-forced-aligner
    ```
2. Download acoustic-model/dictionary
    ```bash
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa
    ```
3. Run script: `bash scripts/3_forced_aligner.py data/GEN_DATA_PATH/`

## Dataset
* Original [MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling](https://aclanthology.org/D18-1547.pdf)
* Improved [MultiWOZ 2.2 : A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines](https://aclanthology.org/2020.nlp4convai-1.13.pdf)
* Implemented using [Huggingface-datasets](https://github.com/huggingface/datasets)


```latex
@inproceedings{budzianowski2018large,
    Author = {Budzianowski, Pawe{\l} and Wen, Tsung-Hsien and Tseng, Bo-Hsiang  and Casanueva, I{\~n}igo and Ultes Stefan and Ramadan Osman and Ga{\v{s}}i\'c, Milica},
    title={MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2018}
}
@inproceedings{zang2020multiwoz,
  title={MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines},
  author={Zang, Xiaoxue and Rastogi, Abhinav and Sunkara, Srinivas and Gupta, Raghav and Zhang, Jianguo and Chen, Jindong},
  booktitle={Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, ACL 2020},
  pages={109--117},
  year={2020}
}
@inproceedings{huggingface-datasets,
    title = "Datasets: A Community Library for Natural Language Processing",
    author = "Lhoest, Quentin  and
      Villanova del Moral, Albert  and
      Jernite, Yacine  and
      Thakur, Abhishek  and
      von Platen, Patrick  and
      Patil, Suraj  and
      Chaumond, Julien  and
      Drame, Mariama  and
      Plu, Julien  and
      Tunstall, Lewis  and
      Davison, Joe  and
      {\v{S}}a{\v{s}}ko, Mario  and
      Chhablani, Gunjan  and
      Malik, Bhavitvya  and
      Brandeis, Simon  and
      Le Scao, Teven  and
      Sanh, Victor  and
      Xu, Canwen  and
      Patry, Nicolas  and
      McMillan-Major, Angelina  and
      Schmid, Philipp  and
      Gugger, Sylvain  and
      Delangue, Cl{\'e}ment  and
      Matussi{\`e}re, Th{\'e}o  and
      Debut, Lysandre  and
      Bekman, Stas  and
      Cistac, Pierric  and
      Goehringer, Thibault  and
      Mustar, Victor  and
      Lagunas, Fran{\c{c}}ois  and
      Rush, Alexander  and
      Wolf, Thomas",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-demo.21",
    pages = "175--184",
    abstract = "The scale, variety, and quantity of publicly-available NLP datasets has grown rapidly as researchers propose new tasks, larger models, and novel benchmarks. Datasets is a community library for contemporary NLP designed to support this ecosystem. Datasets aims to standardize end-user interfaces, versioning, and documentation, while providing a lightweight front-end that behaves similarly for small datasets as for internet-scale corpora. The design of the library incorporates a distributed, community-driven approach to adding datasets and documenting usage. After a year of development, the library now includes more than 650 unique datasets, has more than 250 contributors, and has helped support a variety of novel cross-dataset research projects and shared tasks. The library is available at https://github.com/huggingface/datasets.",
    eprint={2109.02846},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
}
```
