import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from utils.text import prepare_text

venue_transform = {
    'acm sigmod record': "sigmod rec",
    'sigmod record': "sigmod rec",
    'international conference on management of data': "sigmod conf",
    'sigmod conference': "sigmod conf",
    'acm trans . database syst .': "trans",
    'acm transactions on database systems ( tods )': "trans",
    'the vldb journal -- the international journal on very large data bases': "vldb",
    'very large data bases': "vldb",
    'vldb': "vldb",
    'vldb j.': "vldb"
}

table_a = pd.read_csv('data/tableA.csv')
table_b = pd.read_csv('data/tableB.csv')

df_train = pd.read_csv('data/train.csv')
df_train['ltable_id'] = 'A_' + df_train['ltable_id'].astype(str)
df_train['rtable_id'] = 'B_' + df_train['rtable_id'].astype(str)

table_a['id'] = 'A_' + table_a['id'].astype(str)
table_b['id'] = 'B_' + table_b['id'].astype(str)

dataset = pd.concat([table_a, table_b])
dataset['year'] = dataset['year'].fillna(0).astype(int).astype(str).replace({'0': ''})
dataset['venue'] = dataset['venue'].map(venue_transform)
dataset = dataset.fillna('')
dataset['text'] = dataset['title'] + ' ' + dataset['authors'] + ' ' + dataset['authors'] + ' ' + dataset['year']
dataset['text'] = dataset['text'].apply(prepare_text)

ids = dataset['id'].values
texts = dataset['text'].values

documents = [TaggedDocument(doc, [i]) for i, doc in zip(ids, texts)]

model = Doc2Vec(
    documents,
    dm=0,
    vector_size=100,
    window=9,
    min_count=0,
    hs=1,
    negative=0,
    epochs=400,
    workers=8
)

model.save('models/doc2vec-100-min-count-0.model')
