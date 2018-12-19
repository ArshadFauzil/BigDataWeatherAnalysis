from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, col, count
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('DelhiWeather.csv', header = True, inferSchema = True)
# df.printSchema()

df = df.select(' _dewptm', ' _fog',' _pressurem', ' _rain', ' _tempm', \
                ' _thunder', ' _vism', ' _wdird', ' _wspdm', ' _conds')

cols = df.columns

stages = []

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
# df.printSchema()
#
# print(pd.DataFrame(df.take(22), columns=df.columns).transpose())

train, test = df.randomSplit([0.7, 0.3])
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(train)

predictions = lrModel.transform(test)

# predictions.filter(predictions['prediction'] == 0) \
#     .select("label", "prediction") \
#     .orderBy("probability", ascending=False) \
#     .show(n = 10, truncate = 30)

# # trainingSummary = lrModel.summary
# # trainingSummary.roc.show()
# print("areaUnderROC: " + str(trainingSummary.areaUnderROC))
# plt.plot(roc['FPR'], roc['TPR'])
# plt.ylabel('False Positive Rate')
# plt.xlabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()
#
# print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))