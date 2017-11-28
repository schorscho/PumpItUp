from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, StandardScaler, VectorAssembler

import spark_functions


sc = SparkContext("local[3]", "test")
spark = SparkSession(sc)

df = spark_functions.load_data(spark)

df_train, df_test = spark_functions.train_test_split(df)

df_train = spark_functions.prepare_features(df_train)


paymentIndexer = StringIndexer(inputCol='payment', outputCol="payment_indexed").fit(df_train)
schemeManagementIndexer = StringIndexer(inputCol='scheme_management', outputCol="scheme_management_indexed").fit(df_train)
basinIndexer = StringIndexer(inputCol='basin', outputCol="basin_indexed").fit(df_train)
managementIndexer = StringIndexer(inputCol='management', outputCol="management_indexed").fit(df_train)
qualityIndexer = StringIndexer(inputCol='water_quality', outputCol="water_quality_indexed").fit(df_train)
quantityIndexer = StringIndexer(inputCol='quantity', outputCol="quantity_indexed").fit(df_train)
sourceIndexer = StringIndexer(inputCol='source', outputCol="source_indexed").fit(df_train)
extractionTypeIndexer = StringIndexer(inputCol='extraction_type', outputCol="extraction_type_indexed").fit(df_train)
waterpointTypeIndexer = StringIndexer(inputCol='waterpoint_type', outputCol="waterpoint_type_indexed").fit(df_train)

assembler = VectorAssembler(
    inputCols=['latitude', 'longitude', 'gps_height', 'construction_year', 'population', 'payment_indexed', 'scheme_management_indexed', 'basin_indexed', 'management_indexed',
               'water_quality_indexed', 'quantity_indexed', 'source_indexed', 'extraction_type_indexed',
               'waterpoint_type_indexed'],
    outputCol="features")

scaler = StandardScaler(inputCol='features', outputCol='features_scaled', withStd=True, withMean=False)

labelIndexer = StringIndexer(inputCol="status_group", outputCol="label").fit(df_train)

rf = RandomForestClassifier(labelCol='label', featuresCol='features_scaled', seed=42, maxMemoryInMB=2048)
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="status_group_prediction", labels=labelIndexer.labels)

param_grid = ParamGridBuilder()\
    .addGrid(assembler.outputCol, ['features_scaled'])\
    .addGrid(rf.maxDepth, [10])\
    .addGrid(rf.maxBins, [20])\
    .addGrid(rf.minInstancesPerNode, [1])\
    .addGrid(rf.minInfoGain, [0.0])\
    .addGrid(rf.impurity, ['gini'])\
    .addGrid(rf.numTrees, [30])\
    .addGrid(rf.featureSubsetStrategy, ['all'])\
    .build()

pipeline = Pipeline(stages=[paymentIndexer, schemeManagementIndexer, basinIndexer, qualityIndexer, managementIndexer, quantityIndexer, sourceIndexer, extractionTypeIndexer, waterpointTypeIndexer, assembler, labelIndexer, rf, labelConverter])

cross_val = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=5, seed=42)

model = cross_val.fit(df_train)

predictions = model.transform(df_train)

accuracy = evaluator.evaluate(predictions)
print("Training Accuracy = %g" % accuracy)

# Make predictions.
df_test = spark_functions.prepare_features(df_test)
predictions = model.transform(df_test)

accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g" % accuracy)

print(model.bestModel.stages[12])

predictions.show()

from sklearn.metrics import classification_report
predictions = predictions.toPandas()
y_test = predictions['status_group']
y_pred = predictions['status_group_prediction']
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(y_test, y_pred)
print(conf_mx)
sc.stop()
