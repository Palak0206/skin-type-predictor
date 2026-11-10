import re
import spacy

from spellchecker import SpellChecker
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

nlp = spacy.load("en_core_web_sm")

# Preserve these words during stopword removal and spelling correction
KEEP_WORDS = {
    "not", "no", "never", "neither", "nor", "without",
    "very", "extremely", "highly", "slightly", "quite",
    "too", "really", "barely", "almost", "often", "usually",
    "sometimes", "rarely", "always", "frequently", "occasionally",
    "more", "less", "most", "least", "much", "little",
    "high", "medium", "low",
    "oily", "dry", "sensitive", "normal", "combination",
    "skin", "face", "forehead", "cheeks", "pores", "acne",
    "greasy", "itchy", "irritated", "tight", "flaky",
    "hydrated", "dehydrated", "redness", "breakouts"
}

# Explicit, skin-context typo corrections
CUSTOM_CORRECTIONS = {
    "skinn": "skin",
    "sking": "skin",
    "skine": "skin",
    "oiley": "oily",
    "oilness": "oiliness",
    "sensetive": "sensitive",
    "sensetivity": "sensitivity",
    "drynes": "dryness",
    "moisturiser": "moisturizer",
    "moisturiser's": "moisturizer"
}

spell = SpellChecker()
spell.word_frequency.load_words(KEEP_WORDS)
spell.word_frequency.load_words(CUSTOM_CORRECTIONS.values())

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "it's": "it is",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "you're": "you are",
    "they're": "they are",
    "that's": "that is"
}

# 1. Basic preprocessing
def basic_preprocessing(texts):
    results = []

    for text in texts:
        text = str(text).lower().strip()

        for contraction, expanded in CONTRACTIONS.items():
            text = re.sub(
                r"\b" + re.escape(contraction) + r"\b",
                expanded, text
            )

        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        results.append(text)

    return results


# 2. Conservative spelling correction
def spelling_correction(texts):
    results = []

    for text in texts:
        corrected_words = []

        for word in text.split():
            if word in CUSTOM_CORRECTIONS:
                corrected_words.append(CUSTOM_CORRECTIONS[word])

            elif word in KEEP_WORDS or len(word) <= 2:
                corrected_words.append(word)

            else:
                # Only accept a correction for a one-character typo.
                candidate = spell.correction(word)

                if (
                    candidate
                    and candidate != word
                    and spell.unknown([word])
                    and len(word) >= 4
                    and spell.known([candidate])
                    and abs(len(candidate) - len(word)) <= 1
                ):
                    corrected_words.append(candidate)
                else:
                    corrected_words.append(word)

        results.append(" ".join(corrected_words))

    return results


# 3. Stopword removal while preserving intensifiers and negations
def stopword_removal(texts):
    results = []

    for text in texts:
        words = [
            word for word in text.split()
            if word not in STOP_WORDS or word in KEEP_WORDS
        ]
        results.append(" ".join(words))

    return results


# 4. POS-tag-aware lemmatization
def lemmatization(texts):
    results = []

    for doc in nlp.pipe(texts, batch_size=64):
        lemmas = []

        for token in doc:
            if token.is_space or token.is_punct:
                continue

            # Use spaCy's POS-aware lemma
            lemma = token.lemma_.lower()

            lemmas.append(lemma)

        results.append(" ".join(lemmas))

    return results


print("All four preprocessing functions are ready.")