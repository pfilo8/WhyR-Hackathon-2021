import joblib
import metric_learn
import numpy as np
import pandas as pd

from gensim.models.doc2vec import Doc2Vec
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from utils.plotting import result_trace, result_mean_trace

target = 'label'
vector_size = '100-min-count-0'


def get_model_name(algo):
    return str(algo).split('.')[-1][:-2]


def extract_pair(model, item1, item2):
    vec1 = model.docvecs[item1]
    vec2 = model.docvecs[item2]
    return np.array([vec1, vec2])


def get_pairs(dataframe):
    pairs = dataframe.apply(lambda row: extract_pair(model, row['ltable_id'], row['rtable_id']), axis=1).values
    pairs = np.dstack(pairs)
    pairs = np.moveaxis(pairs, 2, 0)
    return pairs


df_train = pd.read_csv('data/train.csv')
df_train['ltable_id'] = 'A_' + df_train['ltable_id'].astype(str)
df_train['rtable_id'] = 'B_' + df_train['rtable_id'].astype(str)

model = Doc2Vec.load(f'models/doc2vec-{vector_size}.model')

cs = [1.0]

for c in cs:
    print(f'Current algorithm: {c}')

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits).split(df_train, df_train[target])

    results_train_acc = []
    results_train_f1 = []
    results_test_acc = []
    results_test_f1 = []

    for idx, (train_idx, test_idx) in enumerate(cv):
        print(f'Iteration: {idx}')
        y_true_train = df_train.iloc[train_idx][target]
        y_true_test = df_train.iloc[test_idx][target]

        pairs_train = get_pairs(df_train.iloc[train_idx])
        pairs_test = get_pairs(df_train.iloc[test_idx])

        x_train = pairs_train[:, 0] - pairs_train[:, 1]
        x_test = pairs_test[:, 0] - pairs_test[:, 1]

        metric_model = SVC(C=c, kernel='rbf')
        metric_model.fit(x_train, y_true_train)
        y_pred_train = metric_model.predict(x_train)
        y_pred_test = metric_model.predict(x_test)

        results_train_acc.append(accuracy_score(y_true_train, y_pred_train))
        results_train_f1.append(f1_score(y_true_train, y_pred_train))
        results_test_acc.append(accuracy_score(y_true_test, y_pred_test))
        results_test_f1.append(f1_score(y_true_test, y_pred_test))

    print('Final train.')
    pairs_train = get_pairs(df_train)
    x_train = pairs_train[:, 0] - pairs_train[:, 1]
    y_train = df_train[target]

    metric_model = SVC(C=c, kernel='rbf')
    metric_model.fit(x_train, y_train)
    joblib.dump(metric_model, f'models/SVM-{vector_size}.model')

    x = np.arange(n_splits)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=['Accuracy', 'F1 Score'])
    fig.add_trace(result_trace(x, results_train_acc, 'Accuracy train', 'blue'), row=1, col=1)
    fig.add_trace(result_mean_trace(x, results_train_acc, 'Accuracy train', 'blue'), row=1, col=1)
    fig.add_trace(result_trace(x, results_test_acc, 'Accuracy test', 'red'), row=1, col=1)
    fig.add_trace(result_mean_trace(x, results_test_acc, 'Accuracy test', 'red'), row=1, col=1)
    fig.add_trace(result_trace(x, results_train_f1, 'F1 train', 'blue'), row=1, col=2)
    fig.add_trace(result_mean_trace(x, results_train_f1, 'F1 train', 'blue'), row=1, col=2)
    fig.add_trace(result_trace(x, results_test_f1, 'F1 train', 'red'), row=1, col=2)
    fig.add_trace(result_mean_trace(x, results_test_f1, 'F1 train', 'red'), row=1, col=2)

    fig.update_layout(yaxis_range=[0, 1])

    fig.write_html(f'outputs/SVM-{vector_size}.html')
