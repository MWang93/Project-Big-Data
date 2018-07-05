# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator


#spark-submit --master local[*] --packages com.databricks:spark-csv_2.10:1.2.0 randomforest.py
#Data Preparation
sc = SparkContext()
sqlContext = SQLContext(sc)
data = sc.textFile('file:/Users/wangmengyuan/Desktop/rr/score.csv').map(lambda l:l.split(','))\
    .map(lambda l:(float(l[0]),float(l[1]),float(l[2]),float(l[3]),float(l[4]),float(l[5]),float(l[6]),float(l[7]),float(l[8]),float(l[9]),float(l[10]),float(l[11]),float(l[12]),float(l[13]),float(l[14]),float(l[15])))
df = sqlContext.createDataFrame(data,["scores","identity","property","room","accommodates","bathrooms","bedrooms","beds","price","guests","people","minnights","numreviews","accuracy","cancellation","reviewspermonth"])

# convert the data to dense vector
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[1:15]),r[0]]).toDF(['features','label'])

# Split the data into training and test sets (40% held out for testing)
data= transData(df)

(training, testing) = data.randomSplit([0.9, 0.1])

# Train a DecisionTree model.
rf = RandomForestRegressor()
model = rf.fit(training)

# Make predictions.
predictions = model.transform(testing)


# Select example rows to display.
predictions.select("label", "features","prediction").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)