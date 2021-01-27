import joblib
import numpy as np
import pandas as pd

from gensim.models.doc2vec import Doc2Vec
from metric_learn import ITML


def extract_pair(model, item1, item2):
    vec1 = model.docvecs[item1]
    vec2 = model.docvecs[item2]
    return np.array([vec1, vec2])


def get_x_pairs(dataframe, model):
    pairs = dataframe.apply(lambda row: extract_pair(model, row['temp_ltable_id'], row['temp_rtable_id']),
                            axis=1).values
    pairs = np.dstack(pairs)
    pairs = np.moveaxis(pairs, 2, 0)
    return pairs


algo_name = 'ITML'
vector_size = '100-min-count-0'

model = Doc2Vec.load(f'models/doc2vec-{vector_size}.model')
model_metric = joblib.load(f'models/{algo_name}-{vector_size}.model')

df_test = pd.read_csv('data/valid.csv')
df_test['temp_ltable_id'] = 'A_' + df_test['ltable_id'].astype(str)
df_test['temp_rtable_id'] = 'B_' + df_test['rtable_id'].astype(str)

pairs = get_x_pairs(df_test, model)
df_test['label'] = (model_metric.predict(pairs) + 1) / 2
df_test = df_test[['ltable_id', 'rtable_id', 'label']]
df_test.to_csv(f'results/submission-{algo_name}-{vector_size}.csv', index=False)
