from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import IndexToString, StringIndexer, StandardScaler, VectorAssembler

import spark_functions


sc = SparkContext("local[3]", "test")
spark = SparkSession(sc)

df = spark_functions.load_data(spark)

df_train, df_test = spark_functions.train_test_split(df)

df_train = spark_functions.prepare_features(df_train)

assembler = VectorAssembler(
    inputCols=['latitude', 'longitude', 'gps_height', 'construction_year'],
    outputCol="features")

scaler = StandardScaler(inputCol='features', outputCol='features_scaled', withStd=True, withMean=False)

labelIndexer = StringIndexer(inputCol="status_group", outputCol="label").fit(df_train)

lr = LogisticRegression(labelCol='label', featuresCol='features_scaled', standardization=False, family='multinomial')
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="status_group_prediction", labels=labelIndexer.labels)

param_grid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1])\
    .addGrid(lr.elasticNetParam, [1.0])\
    .build()

pipeline = Pipeline(stages=[assembler, scaler, labelIndexer, lr, labelConverter])

cross_val = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=5, seed=42)

model = cross_val.fit(df_train)

# Make predictions.
df_test = spark_functions.prepare_features(df_test)
predictions = model.transform(df_test)

predictions.show()

accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)

print(model.bestModel.stages[3])

sc.stop()
