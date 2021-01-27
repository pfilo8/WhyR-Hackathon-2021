import re
import string

from .config import VENUE_DICT, VENUE_ENCODE


def text_preproc(x):
    x = x.lower()
    x = ' '.join([word for word in x.split(' ') if word not in string.punctuation])
    x = re.sub(r'#\s?', '', x)
    x = re.sub(r'\s&\s?', ' ', x)
    x = re.sub(r';\s?', ' ', x)
    x = x.encode('ascii', 'ignore').decode()
    x = re.sub(r'https*\S+', ' ', x)
    x = re.sub(r'@\S+', ' ', x)
    x = re.sub(r'\'\w+', '', x)
    x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
    x = re.sub(r'\s+[0-9]{3}\s+', '', x)
    x = re.sub(r'\s{2,}', ' ', x)
    return x


def encode_venues(df, table_type='A'):
    for index, row in df.iterrows():
        if table_type == 'B':
            for key, value in VENUE_DICT.items():
                # print(row.title.replace(value, key))
                df.title[index] = df.title[index].replace(value, key)
        for key_1, value_1 in VENUE_ENCODE.items():
            df.title[index] = df.title[index].replace(key_1, value_1)
    return df


def preprocess_table(df, table_type='A'):
    df.year = df.year.astype(str)
    df.venue = df.venue.astype(str)
    df.authors = df.authors.astype(str)
    df.year = df.year.str.replace(r'\.0$', '')
    df['title'] = df[['title', 'authors', 'venue', 'year']].agg(' '.join, axis=1).str.replace(r'nan', '')
    # print(df['title'])
    df = encode_venues(df, table_type)
    titles_preprocessed = df['title'].apply(text_preproc)
    return titles_preprocessed
