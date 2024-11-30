from google.colab import drive
drive.mount('/content/drive')

!apt-get install openjdk-8-jdk-headless



!tar -xvf /content/drive/MyDrive/spark-3.4.0-bin-hadoop3.tgz


import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.4.0-bin-hadoop3"



!pip install findspark

import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession
# SparkSession 생성 전에 설정을 적용합니다. Java heap space 문제를 해결하기 위함..
spark = SparkSession.builder \
    .appName("MySparkApp") \
    .config("spark.executor.memory", "20g") \
    .config("spark.driver.memory", "20g") \
    .getOrCreate()


import pyspark
spark_version = pyspark.__version__
print("Apache Spark 버전 확인: " + spark_version)


from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import VectorAssembler #TO FIND OUT THE SUBJECT OF THE ARTICLE
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
from pyspark.ml.feature import HashingTF #TO figure out the subject of the article in Natural language not numeric
from pyspark.sql.functions import when



#LOAD THE FILES FROM COLAB FILE DIRECTORY
#Prepared 3 different datasets are loaded

spark = SparkSession.builder.appName("HW").getOrCreate()
data_set=spark.read.format("csv").load("/content/drive/MyDrive/train (2).csv", header=True, sep=";", inferSchema=True,on_bad_lines='skip')

# Replace null values in the "text" column with empty strings before tokenization
data_set = data_set.withColumn("text", when(col("text").isNull(), "").otherwise(col("text")))


#ready to tokenizer text into words
tokenizer=Tokenizer(inputCol="text",outputCol="words")

#tokenized trained data
wordsData=tokenizer.transform(data_set)

#Removing stop words of tokenized trained data
remover=StopWordsRemover(inputCol="words",outputCol="filtered")
wordsData=remover.transform(wordsData)

#show the tokenized, stop words-removed trained_data (토큰화 되고 스탑월즈가 삭제된 트레인 데이터를 일단 출력해봅니다..)
wordsData.show()

#TF- IDF( To find out the subject of articles)

#TF AND IDF SETTINGS
hashingTF=HashingTF(inputCol="filtered",outputCol="rawFeatures")
tfData=hashingTF.transform(wordsData)

idf=IDF(inputCol="rawFeatures",outputCol="features")
idfModel=idf.fit(tfData)

#TF-IDF ED DATA
tf_idfData=idfModel.transform(tfData)


# Filter rows where _c0
#tf_idfData = tf_idfData.filter(tf_idfData["_c0"] <500)

tf_idfData.select("_c0","features").show()

#TF-IDF의 ID결과값을 문자로 나타내게 하여 각 기사의 주제를 파악하고자 하였다.

import numpy as np
from pyspark.ml.feature import CountVectorizer
tf_idf_df = tf_idfData.select("features")

# 첫 번째 문서의 벡터를 NumPy 배열로 변환
vector = tf_idf_df.first().features.toArray()

# 가장 큰 값의 인덱스 찾기
max_index = np.argmax(vector)

cv = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")
cvModel = cv.fit(wordsData)  # CountVectorizer를 데이터에 fit
vocabulary = cvModel.vocabulary


from pyspark.sql.types import StringType # Change String to StringType

# Define a UDF to find the word with the highest TF-IDF value in each row
def find_max_tfidf_word(features):
    vector = features.toArray()
    max_index = np.argmax(vector)
    try:
        word = vocabulary[max_index]
    except IndexError:
        word = "N/A"  # Handle cases where max_index is out of vocabulary bounds
    return word

# Register the UDF
find_max_tfidf_word_udf = udf(find_max_tfidf_word, StringType())

# Apply the UDF to the DataFrame
result_df = tf_idfData.withColumn("max_tfidf_word", find_max_tfidf_word_udf("features"))

# Show the results
result_df.select("_c0", "max_tfidf_word").show()


#여기까지가 주제찾기 코드 차후에 이용하도록 합시다.

tf_idfData = tf_idfData.filter(tf_idfData["_c0"] <2000) #to deal with Too much resources errors

#DECISION TREE FROM NOW.
#Fit on whole dataset io include all
labelIndexer=StringIndexer(inputCol="label",outputCol="indexedLabel", handleInvalid="skip").fit(tf_idfData)

#Automatically identify categorical features, and index them...
featureIndexer=VectorIndexer(inputCol="features",outputCol="indexedFeatures",maxCategories=6).fit(tf_idfData)

train_data, test_data= tf_idfData.randomSplit([0.8,0.2],seed=42)

dt=DecisionTreeClassifier(labelCol="indexedLabel",featuresCol="indexedFeatures") #maxBins=1000 test code.. to figure out the errors..

pipeline=Pipeline(stages=[labelIndexer,featureIndexer,dt])

model=pipeline.fit(train_data)

predictions=model.transform(test_data)

#Select example rows to display
predictions.select("prediction","indexedLabel","features").show(5)


evaluator=MulticlassClassificationEvaluator(labelCol="indexedLabel",predictionCol="prediction",metricName="accuracy")
accuracy=evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
treeModel=model.stages[2]
print(treeModel)
