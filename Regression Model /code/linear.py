# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
#spark-submit --master local[*] --packages com.databricks:spark-csv_2.10:1.2.0 linear.py
# Data Preparation
sc = SparkContext()
sqlContext = SQLContext(sc)
# Load training data
data = sc.textFile('file:/Users/wangmengyuan/Desktop/Bigdata/score.csv').map(lambda l:l.split(','))\
    .map(lambda l:(l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8],l[9],l[10],l[11],l[12],l[13],l[14],l[15]))
df = sqlContext.createDataFrame(data,["scores","identity","property","room","accommodates","bathrooms","bedrooms","beds","price","guests","people","minnights","numreviews","accuracy","cancellation","reviewspermonth"])
count=df.count()
list=df.collect()
file = open('linear.txt','w+')
for row in list:
    file.write('%s 1:%s 2:%s 3:%s 4:%s 5:%s 6:%s 7:%s 8:%s 9:%s 10:%s 11:%s 12:%s 13:%s 14:%s 15:%s\n'\
                          % (row['scores'], row['identity'], row['property'], row['room'] \
                                 , row['accommodates'], row['bathrooms'], row['bedrooms'] \
                                 , row['beds'], row['price'], row['guests'], row['people'] \
                                 , row['minnights'], row['numreviews'], row['accuracy'] \
                                 , row['cancellation'], row['reviewspermonth']))

file.close()

# Fitthemodel
data=sqlContext.read.format("libsvm")\
    .load("file:/Users/wangmengyuan/Desktop/rr/linear.txt")

# Split the data into training and test sets (40% held out for testing)
(training, testing) = data.randomSplit([0.7, 0.3])

lr=LinearRegression(maxIter=10,regParam=0.3,elasticNetParam=0.8)
lrModel=lr.fit(training)
# Print coefficients and intercept for linear regression
print("Coefficients:"+str(lrModel.coefficients))
print("Intercept:"+str(lrModel.intercept))

# Prediction
predictions = lrModel.transform(testing)
#predictions.write.format('com.databricks.spark.csv').save('file:/Users/wangmengyuan/Desktop/rr/prediction.csv')
predictions.select("label", "features","prediction").show(5)
# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
RMSE = evaluator.evaluate(predictions)
print("Model: Root Mean Squared Error = " + str(RMSE))
