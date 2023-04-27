from nltk.corpus import cmudict

WORD = cmudict.dict()

"""
Source: https://datascience.stackexchange.com/questions/23376/how-to-get-the-number-of-syllables-in-a-word
Alternative that is not used: https://eayd.in/?p=232 
"""


def number_of_syllables(word):
    if word not in WORD:
        return 0
    # n = [len(list(y for y in x if y[-1].isdigit())) for x in WORD[word.lower()]]
    n = [len(list(y for y in x if y[-1].isdigit())) for x in WORD[word.lower()]][0]
    return n


if __name__ == "__main__":
    pass

    word = "bajs"
    ns = number_of_syllables(word)
    print(f"{word} -> {ns}")
