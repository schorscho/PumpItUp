import os

from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler


PROJECT_ROOT_DIR = "/Users/gopora/MyStuff/Dev/Workspaces/sandbox/PumpItUp"
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
TRAINING_SET_VALUES = "pump_it_up_training_set_values.csv"
TRAINING_SET_LABELS = "pump_it_up_training_set_labels.csv"
TEST_SET_VALUES = "pump_it_up_test_set_values.csv"


def load_data(spark):
    df_values = spark.read.csv(path=os.path.join(DATA_DIR, TRAINING_SET_VALUES), header=True, inferSchema=True)
    df_labels = spark.read.csv(path=os.path.join(DATA_DIR, TRAINING_SET_LABELS), header=True, inferSchema=True)

    df_join = df_values.join(df_labels, 'id')

    return df_join


def train_test_split(df_join):
    df_train, df_test = df_join.randomSplit(weights=[0.8, 0.2], seed=42)

    return df_train, df_test


def prepare_features(df):
    prepared = df.fillna(value='None', subset=['funder', 'installer', 'scheme_management'])

    #prepared = prepared.replace(to_replace=0, value=1950, subset='construction_year')

    #preparation of longitude

    #preparation of latitude

    #preparation of scheme_management

    return prepared



