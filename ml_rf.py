import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nsga2.rf_utils import *
import os

os.makedirs('./data/train_val_test/', exist_ok=True)



def read_data(df_name):
    df = pd.read_csv('/after_process_to_use_compas.csv')
    return df

def score_text(v):
    return {'Low': 0, 'Medium': 1}.get(v, 2)

def get_matrices(seed):
    df = pd.read_csv("after_process_to_use_compas.csv",
                     engine = 'python', 
                     sep = ",",
                     encoding='utf-8')
    target_column = 'recid_use'
    train_column = ['sex', 'age', 'p_felony_count_person', 'p_misdem_count_person',
                'p_age_first_offense', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                'priors_count', 'main_crime_type']
    
    X = df[train_column].copy()
    y = df[target_column].copy()
 
    X['sex'] = X['sex'].map({'Male': 0, 'Female': 1})
    X['main_crime_type'] = X['main_crime_type'].map({'F':1, 'M':0})
    X['main_crime_type'] = X['main_crime_type'].astype(int)
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

def write_train_val_test(seed, X_train, X_val, X_test, y_train, y_val, y_test, group='default'):
    train = X_train.copy()
    val = X_val.copy()
    test = X_test.copy()

    train['y'] = y_train.tolist()
    train.to_csv(f'./data/train_val_test/{group}_train_seed_{seed}.csv', index=False)
    val['y'] = y_val.tolist()
    val.to_csv(f'./data/train_val_test/{group}_val_seed_{seed}.csv', index=False)
    test['y'] = y_test.tolist()
    test.to_csv(f'./data/train_val_test/{group}_test_seed_{seed}.csv', index=False)

def get_group_matrices(seed, group: str):
    """
    group classification: 'F'(Felony group) or 'M'(Misdemeanor group) 
    """
    df = pd.read_csv("after_process_to_use_compas.csv", engine='python',
                     sep=",", encoding='utf-8')
    
    df = df[df['main_crime_type'] == group].copy()
    
    target_column = 'recid_use'
    feature_cols = [
        'sex', 'age', 'p_felony_count_person', 'p_misdem_count_person',
        'p_age_first_offense', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count'
    ]
    X = df[feature_cols].copy()
    y = df[target_column].copy()

    X['sex'] = X['sex'].map({'Male': 0, 'Female': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, random_state=seed, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def write_train_val_tes_group(seed, X_train, X_val, X_test, y_train, y_val, y_test, group: str):
    for split_name, X_, y_ in [
        ('train', X_train, y_train),
        ('val',   X_val,   y_val),
        ('test',  X_test,  y_test),
    ]:
        df_out = X_.copy()
        df_out['y'] = y_.tolist()
        path = f'./data/train_val_test/{group}_{split_name}_seed_{seed}.csv'
        df_out.to_csv(path, index=False)

def get_matrices_felony(seed):
    return get_group_matrices(seed, 'F')

def get_matrices_misdemeanor(seed):
    return get_group_matrices(seed, 'M')

def train_model(df_name, seed, **features):
    train = pd.read_csv(f'./data/train_val_test/{df_name}_train_seed_{seed}.csv')
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]

    params = {
        'n_estimators':      features.get('n_estimators', 100),
        'criterion':         'gini' if features['criterion']<=0.5 else 'entropy',
        'max_depth':         features['max_depth'],
        'min_samples_split': features['min_samples_split'],
        'max_leaf_nodes':    features['max_leaf_nodes'],
        'class_weight':      'balanced' if features['class_weight']==1 else None,
        'random_state':      seed,
        'n_jobs':            -1
    }
    clf = RandomForestClassifier(**params)
    learner = clf.fit(X_train, y_train)
    return learner

def save_model(learner, dataset_name, seed, variable_name,
               num_of_generations, num_of_individuals, individual_id):
    import pickle, os
    path = f'./results/models/{dataset_name}/'
    os.makedirs(path, exist_ok=True)
    filename = (f'{path}model_{dataset_name}_seed_{seed}_'
                f'gen_{variable_name}_indiv_{num_of_generations}_'
                f'{num_of_individuals}_id_{individual_id}.sav')
    pickle.dump(learner, open(filename, 'wb'))

def val_model(df_name, learner, seed):
    #Return validation features, labels, and predicted probabilities for class 1.
    val = pd.read_csv(f'./data/train_val_test/{df_name}_val_seed_{seed}.csv')
    X_val, y_val = val.iloc[:,:-1], val.iloc[:,-1]
    prob = learner.predict_proba(X_val)[:,1]
    return X_val, y_val, prob

def test_model(df_name, learner, seed):
    test = pd.read_csv(f'./data/train_val_test/{df_name}_test_seed_{seed}.csv')
    X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]
    prob = learner.predict_proba(X_test)[:,1]
    return X_test, y_test, prob



