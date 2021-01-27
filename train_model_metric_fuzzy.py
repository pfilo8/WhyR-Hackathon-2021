import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from utils.config import OPTIMAL_THRESHOLD
from utils.preprocess_fuzzy import preprocess_table
from utils.plotting import result_trace, result_mean_trace
from models.FuzzyMatcher import FuzzyMatcher


def prepare_train_table(df, titles_A, titles_B):
    train_set_titles = []
    for index, row in df.iterrows():
        train_set_titles.append((titles_A[row['ltable_id']], titles_B[row['rtable_id']], row['label']))
    return pd.DataFrame(train_set_titles, columns=[0, 1, 'label'])


df_train = pd.read_csv('data/train.csv')
df_A = pd.read_csv('data/tableA.csv')
df_B = pd.read_csv('data/tableB.csv')

df_A = preprocess_table(df_A, table_type='A')
df_B = preprocess_table(df_B, table_type='B')
train_table = prepare_train_table(df_train, df_A, df_B)

n_splits = 10
cv = StratifiedKFold(n_splits=n_splits).split(train_table, train_table['label'])
results_acc_train = []
results_f1_train = []
results_acc_test = []
results_f1_test = []

for idx, (train_idx, test_idx) in enumerate(cv):
    X_train = train_table.loc[train_idx, :]
    X_test = train_table.loc[test_idx, :]
    y_true_train = train_table.loc[train_idx, 'label']
    y_true_test = train_table.loc[test_idx, 'label']
    y_pred_train = FuzzyMatcher().predict(X_train[0], X_train[1], OPTIMAL_THRESHOLD)
    y_pred_test = FuzzyMatcher().predict(X_test[0], X_test[1], OPTIMAL_THRESHOLD)
    results_acc_train.append(accuracy_score(y_true_train, y_pred_train))
    results_acc_test.append(accuracy_score(y_true_test, y_pred_test))
    results_f1_train.append(f1_score(y_true_train, y_pred_train))
    results_f1_test.append(f1_score(y_true_test, y_pred_test))

x = np.arange(n_splits)

fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                    subplot_titles=['Accuracy', 'F1 Score'])
fig.add_trace(result_trace(x, results_acc_train, 'Accuracy train', 'blue'), row=1, col=1)
fig.add_trace(result_mean_trace(x, results_acc_train, 'Accuracy train', 'blue'), row=1, col=1)
fig.add_trace(result_trace(x, results_acc_test, 'Accuracy test', 'red'), row=1, col=1)
fig.add_trace(result_mean_trace(x, results_acc_test, 'Accuracy test', 'red'), row=1, col=1)
fig.add_trace(result_trace(x, results_f1_train, 'F1 train', 'blue'), row=1, col=2)
fig.add_trace(result_mean_trace(x, results_f1_train, 'F1 train', 'blue'), row=1, col=2)
fig.add_trace(result_trace(x, results_f1_test, 'F1 test', 'red'), row=1, col=2)
fig.add_trace(result_mean_trace(x, results_f1_test, 'F1 train', 'red'), row=1, col=2)
fig.update_layout(yaxis_range=[0.9, 1])
fig.write_html(f'outputs/fuzzy-cv_results.html')
