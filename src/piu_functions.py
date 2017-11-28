# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report


# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# Where to save the figures
PROJECT_ROOT_DIR = "/Users/gopora/MyStuff/Dev/Workspaces/PumpItUp"
IMAGES_DIR = os.path.join(PROJECT_ROOT_DIR, "images")
DATASETS_DIR = os.path.join(PROJECT_ROOT_DIR, "datasets")
TRAINING_SET_VALUES = "pump_it_up_training_set_values.csv"
TRAINING_SET_LABELS = "pump_it_up_training_set_labels.csv"
TEST_SET_VALUES = "pump_it_up_test_set_values.csv"


def save_fig(fig_id, tight_layout = True):
    if not os.path.isdir(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    path = os.path.join(IMAGES_DIR, fig_id + ".png")

    print("Saving figure ...", fig_id)
    
    if tight_layout:
        plt.tight_layout()
    
    plt.savefig(path, format = 'png', dpi = 300)
    

def fetch_pump_it_up_data():
    if not os.path.isdir(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
        
    print("Fetching training set values ...")

    root_url  = "https://s3.amazonaws.com/drivendata/data/7/public/"
    file_url  = root_url + "4910797b-ee55-40a7-8668-10efd5c1b960.csv"
    file_path = os.path.join(DATASETS_DIR, TRAINING_SET_VALUES)

    urllib.request.urlretrieve(file_url, file_path)
    
    print("Fetching training set labels ...")

    file_url  = root_url + "0bf8bc6e-30d0-4c50-956a-603fc693d966.csv"
    file_path = os.path.join(DATASETS_DIR, TRAINING_SET_LABELS)

    urllib.request.urlretrieve(file_url, file_path)

    print("Fetching test set values ...")

    file_url  = root_url + "702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv"
    file_path = os.path.join(DATASETS_DIR, TEST_SET_VALUES)

    urllib.request.urlretrieve(file_url, file_path)    
 
    
def load_pump_it_up_training_set_values():
    return pd.read_csv(os.path.join(DATASETS_DIR, TRAINING_SET_VALUES))


def load_pump_it_up_training_set_labels():
    return pd.read_csv(os.path.join(DATASETS_DIR, TRAINING_SET_LABELS))


def load_pump_it_up_test_set_values():
    return pd.read_csv(os.path.join(DATASETS_DIR, TEST_SET_VALUES))   

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    

def piu_load_data():
    trv = load_pump_it_up_training_set_values()
    trl = load_pump_it_up_training_set_labels()
    #tesv = load_pump_it_up_test_set_values()

    return trv, trl

def piu_prepare_labels(pump_labels):
    return pump_labels["status_group"].replace(["functional", "functional needs repair", "non functional"], [2, 1, 0])
   
    
def piu_prepare_values(pump_values):
    num_pipeline = Pipeline([
            ('selector', DataFrameSelector(["longitude", "latitude", "gps_height", "construction_year"])),
            ('std_scaler', StandardScaler()),
        ])
    
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(["funder"])),
        ('label_binarizer', LabelBinarizer()),
    ])
    
    cat_pipeline2 = Pipeline([
        ('selector', DataFrameSelector(["installer"])),
        ('label_binarizer', LabelBinarizer()),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
            ("cat_pipeline2", cat_pipeline2),
        ])

    pump_values["funder"].fillna("_other", inplace=True)
    pump_values["installer"].fillna("_other", inplace=True)
    pump_values["construction_year"].replace(to_replace=0, value=1950, inplace=True)
    
    return full_pipeline.fit_transform(pump_values)


# later to be stratified by location
def piu_train_test_split(X, y):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
 
    for train_index, test_index in split.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

    return X_train, y_train, X_test, y_test

def piu_print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1', 'class 2']))
    
    correct = (y_true == y_pred)
    
    print("Classification Rate: ", len(correct[correct]) / len(y_true))
