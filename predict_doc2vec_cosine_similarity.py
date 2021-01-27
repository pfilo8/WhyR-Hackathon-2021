import pandas as pd

from gensim.models.doc2vec import Doc2Vec
from scipy.spatial import distance


def calculate_cosine_similarity(u, v):
    return 1 - distance.cosine(u, v)


def predict(model, item1, item2, threshold=0.5):
    vec1 = model.docvecs[item1]
    vec2 = model.docvecs[item2]
    return int(calculate_cosine_similarity(vec1, vec2) > threshold)


vector_size = '100-better-data-window-9'

model = Doc2Vec.load(f'models/doc2vec-{vector_size}.model')

threshold = 0.7

df_test = pd.read_csv('data/valid.csv')
df_test['temp_ltable_id'] = 'A_' + df_test['ltable_id'].astype(str)
df_test['temp_rtable_id'] = 'B_' + df_test['rtable_id'].astype(str)

df_test['label'] = df_test.apply(
    lambda row: predict(model, row['temp_ltable_id'], row['temp_rtable_id'], threshold),
    axis=1
)

df_test = df_test[['ltable_id', 'rtable_id', 'label']]
df_test.to_csv(f'results/submission-{vector_size}.csv', index=False)
