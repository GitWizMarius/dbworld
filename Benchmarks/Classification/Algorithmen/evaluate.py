import pickle
import pandas

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Declare Pickle Path
pickles_path = '../Pickles/Models/'

# Declare which Pickles to evaluate

list_subjects = [
    'sum_model_gbm_subject.pickle',
    'sum_model_rf_subject.pickle',
    'sum_model_mlr_subject.pickle',
    'sum_model_mnb_subject.pickle',
    'sum_model_svm_subject.pickle',
]

list_bodys = [
    'sum_model_gbm_body.pickle',
    'sum_model_rf_body.pickle',
    'sum_model_mlr_body.pickle',
    'sum_model_mnb_body.pickle',
    'sum_model_svm_body.pickle',
]

list_boths = [
    'sum_model_gbm_both.pickle',
    'sum_model_rf_both.pickle',
    'sum_model_mlr_both.pickle',
    'sum_model_mnb_both.pickle',
    'sum_model_svm_both.pickle',
]


def get_comparison(list):
    df_summary = pandas.DataFrame()

    for pickle_ in list:
        path = pickles_path + pickle_
        with open(path, 'rb') as data:
            df = pickle.load(data)

        df_summary = df_summary.append(df)

    df_summary = df_summary.reset_index().drop('index', axis=1)
    print(df_summary)

if __name__ == '__main__':
    #get_comparison(list_subjects)
    #get_comparison(list_bodys)
    get_comparison(list_boths)

