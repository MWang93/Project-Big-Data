import sys
import re
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Normalizer
from pyspark.mllib.linalg import Vectors
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

#spark-submit --master local[*] --packages com.databricks:spark-csv_2.10:1.2.0 cluster.py

sc = SparkContext()
sqlContext = SQLContext(sc)
text = sc.textFile('file:/Users/wangmengyuan/Desktop/rr/listings.txt').map(lambda l:l.split('\t'))\
	.map(lambda l: (l[0],l[1]))
df = sqlContext.createDataFrame(text,["houseid", "description"])
tokenizer = Tokenizer(inputCol="description", outputCol="tokens")
tokenized = tokenizer.transform(df).cache()
remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
stopWordsRemoved_df = remover.transform(tokenized).cache()
hashingTF = HashingTF (inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures=200)
tfVectors = hashingTF.transform(stopWordsRemoved_df).cache()    
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
idfModel = idf.fit(tfVectors)
tfIdfVectors = idfModel.transform(tfVectors).cache()
normalizer = Normalizer(inputCol="features", outputCol="normFeatures")
l2NormData = normalizer.transform(tfIdfVectors)
kmeans = KMeans().setK(10).setMaxIter(20)
km_model = kmeans.fit(l2NormData)
clustersTable = km_model.transform(l2NormData)

#save to hdfs
df1 = clustersTable[['houseid','prediction']]
#df1.select('houseid', 'prediction').write.format('com.databricks.spark.csv').save('cluster.csv')
df1.select('houseid', 'prediction').show(20)
sc.stop()

