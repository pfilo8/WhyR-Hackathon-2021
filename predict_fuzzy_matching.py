import pandas as pd

from utils.preprocess_fuzzy import preprocess_table
from utils.config import OPTIMAL_THRESHOLD
from models.FuzzyMatcher import FuzzyMatcher


def prepare_test_table(df, titles_A, titles_B):
    test_set_titles = []
    for index, row in df.iterrows():
        test_set_titles.append((titles_A[row['ltable_id']], titles_B[row['rtable_id']]))
    return pd.DataFrame(test_set_titles)


df_test = pd.read_csv('data/valid.csv')
df_A = pd.read_csv('data/tableA.csv')
df_B = pd.read_csv('data/tableB.csv')

df_A = preprocess_table(df_A, table_type='A')
df_B = preprocess_table(df_B, table_type='B')
test_table = prepare_test_table(df_test, df_A, df_B)

df_test['label'] = FuzzyMatcher().predict(test_table[0], test_table[1], OPTIMAL_THRESHOLD)
df_test.to_csv(f'results/submission-fuzzy.csv', index=False)
