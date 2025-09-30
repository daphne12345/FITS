import time
from ConfigSpace import ConfigurationSpace, Float, Integer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
import numpy as np
import random
from argparse import ArgumentParser
from Data_Preparation import prepare_data


class Tuner:
    """Tuner to tune the model with Smac.
    """
    def __init__(self, X_train, X_val, y_train, y_val, seed):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.seed=seed

        
        
    @property
    def configspace(self) -> ConfigurationSpace:
        """
        Build Configuration Space which defines the hyperparameters and their ranges.
        :return: ConfigurationSpace
        """
        cs = ConfigurationSpace(seed=self.seed)
        cs.add([
            Float("C", (0.1, 10), default=1, log=True),
            Integer("n_gram_max", (1,4), default=2, log=False)
        ])
        return cs

    def train(self, config, seed=0):
        """
        Trains and evaluates the tfidfi-svm pipeline witht he given configuration.
        :param config: hyperparameter configuration
        :return: perfromance error
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english',
                                      ngram_range=(1,config['n_gram_max']))),
            ('svm', OneVsRestClassifier(SVC(
                kernel='linear',
                C=config['C'],
                random_state=seed
            )))
        ])

        pipeline.fit(self.X_train, self.y_train)
        y_val_pred = pipeline.predict(self.X_val)

        time.sleep(5)

        score = f1_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        return 1- score

def hpo(X_train, X_val, y_train, y_val):
    """
    Performs hyperparameter optimization with Smac.
    :param X_train: x_train
    :param X_val: X_val
    :param y_train: y_train
    :param y_val: y_val

    :return: best configuration
    """
    tuner = Tuner(X_train, X_val, y_train, y_val, args.seed)

    scenario = Scenario(
        tuner.configspace,
        n_trials=1000,
        deterministic=True,
        seed=args.seed
    )

    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)

    smac = HPOFacade(
        scenario,
        tuner.train,
        initial_design=initial_design,
        overwrite=False
    )

    return smac.optimize()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--seed', default=0, type=int)
    args = parser.parse_args()

    print('seed', args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
   
    df, df_train, df_val, df_test = prepare_data()
    X_train = df_train['windowed_3'].values
    y_train = df_train.iloc[:, 1:].to_numpy().astype(int).tolist()
    X_val = df_val['windowed_3'].values
    y_val = df_val.iloc[:, 1:].to_numpy().astype(int).tolist()
    X_test = df_test['windowed_3'].values
    y_test = df_test.iloc[:, 1:].to_numpy().astype(int).tolist()

    incumbent = hpo()
    print(incumbent)
    
    # Train and test the final model with the best found configuration
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english',
                                  ngram_range=(1, incumbent['n_gram_max']))),
        ('svm', OneVsRestClassifier(SVC(
            kernel='linear',
            C=incumbent['C'],
            random_state=args.seed
        )))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Print the classification report and F1-score
    print("Classification Report:\n", classification_report(y_test, y_pred))

    print('f1_weighted', f1_score(y_test, y_pred, average='weighted', zero_division=0))
    print('f1_macro', f1_score(y_test, y_pred, average='macro', zero_division=0))
    print('f1_micro', f1_score(y_test, y_pred, average='micro', zero_division=0))
    print('precision_weighted', precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print('recall_weighted', recall_score(y_test, y_pred, average='weighted', zero_division=0))
