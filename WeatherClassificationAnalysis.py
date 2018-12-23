from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, col, count
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('DelhiWeather.csv', header = True, inferSchema = True)
# df.printSchema()

df = df.select(' _dewptm', ' _fog',' _pressurem', ' _rain', ' _tempm', \
                ' _thunder', ' _vism', ' _wdird', ' _wspdm', ' _conds')

cols = df.columns

stages = []

# df.groupBy(" _conds") \
#     .count() \
#     .orderBy(col("count").desc()) \
#     .show()

df_stats = df.select(
    _mean(col(' _vism')).alias('mean_vism'),
    _mean(col(' _wdird')).alias('mean_wdird'),
    _mean(col(' _wspdm')).alias('mean_wspdm'),).collect()

mean_vism = df_stats[0]['mean_vism']
mean_wdird = df_stats[0]['mean_wdird']
mean_wspdm = df_stats[0]['mean_wspdm']

df = df.fillna({' _vism': mean_vism})
df = df.fillna({' _wdird': mean_wdird})
df = df.fillna({' _wspdm': mean_wspdm})


# print(mean_vism , mean_wdird , mean_wspdm)

# numeric_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1] == 'float'or t[1] == 'double']
# # print(df.select(numeric_features).describe().toPandas().transpose())
#
# numeric_data = df.select(numeric_features).toPandas()

label_stringIdx = StringIndexer(inputCol = ' _conds', outputCol = 'label').setHandleInvalid("skip")
stages += [label_stringIdx]

vectorAssembler = VectorAssembler(inputCols = [' _dewptm', ' _fog',' _pressurem', ' _rain', ' _tempm', \
                                               ' _thunder', ' _vism', ' _wdird', ' _wspdm'], \
                                                outputCol = 'features').setHandleInvalid("skip")

stages += [vectorAssembler]

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()
#
# print(pd.DataFrame(df.take(22), columns=df.columns).transpose())

train, test = df.randomSplit([0.7, 0.3])
# print("Training Dataset Count: " + str(train.count()))
# print("Test Dataset Count: " + str(test.count()))

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 10, \
                            maxBins = 32)

#CROSS - VALIDATION
# paramGrid = (ParamGridBuilder()
#              .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
#              .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
# #            .addGrid(model.maxIter, [10, 20, 50]) #Number of iterations
# #            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
#              .build())
#
# evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
#
# # Create 7-fold CrossValidator
# cv = CrossValidator(estimator=lr, \
#                     estimatorParamMaps=paramGrid, \
#                     evaluator=evaluator, \
#                     numFolds=5)

lrModel = lr.fit(train)
# cvModel = cv.fit(train)
# rfModel = rf.fit(train)

predictions = lrModel.transform(test)
# predictions = cvModel.transform(test)
# predictions = rfModel.transform(test)

results = predictions.select(['prediction', 'label'])
predictionAndLabels = results.rdd

metrics = MulticlassMetrics(predictionAndLabels)

cm = metrics.confusionMatrix().toArray()
accuracy = (cm[0][0]+cm[1][1])/cm.sum()
precision = (cm[0][0])/(cm[0][0]+cm[1][0])
recall = (cm[0][0])/(cm[0][0]+cm[0][1])
f1score = 2*((precision*recall)/(precision+recall))

# print(evaluator.evaluate(predictions))
print("Classifier: accuracy, precision, recall, f1score", accuracy, precision, recall, f1score)