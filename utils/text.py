import re
import string
import unidecode


def convert_latin_to_unicode(t):
    return re.sub(r' &#(\d+); ', lambda m: chr(int(m.group(1))), t)


def preprocess_text(t):
    text = convert_latin_to_unicode(t)
    text = text.lower()
    for punct in string.punctuation:
        text = text.replace(punct, '')
    text = unidecode.unidecode(text)
    text = ' '.join([el for el in text.split() if len(el) > 1])
    return text


def split_text(t):
    return t.split(' ')


def prepare_text(t):
    t = preprocess_text(t)
    t = split_text(t)
    return t
