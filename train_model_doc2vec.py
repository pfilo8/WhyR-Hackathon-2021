import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from utils.text import prepare_text

table_a = pd.read_csv('data/tableA.csv')
table_b = pd.read_csv('data/tableB.csv')

df_train = pd.read_csv('data/train.csv')
df_train['ltable_id'] = 'A_' + df_train['ltable_id'].astype(str)
df_train['rtable_id'] = 'B_' + df_train['rtable_id'].astype(str)

dataset_a = table_a['title'].apply(prepare_text).values.tolist()
ids_a = [f'A_{el}' for el in table_a['id']]
dataset_b = table_b['title'].apply(prepare_text).values.tolist()
ids_b = [f'B_{el}' for el in table_b['id']]
dataset = [*dataset_a, *dataset_b]
ids = [*ids_a, *ids_b]

assert len(dataset_a) == len(ids_a)
assert len(dataset_b) == len(ids_b)
assert len(dataset) == len(ids)

documents = [TaggedDocument(doc, [i]) for i, doc in zip(ids, dataset)]

model = Doc2Vec(
    documents,
    vector_size=100,
    window=2,
    min_count=1,
    hs=1,
    negative=0,
    epochs=400,
    workers=8
)

model.save('models/doc2vec.model')
