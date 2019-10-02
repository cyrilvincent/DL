import os

from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, StringIndexer
from pyspark.sql import Row

def load_dataframe(path):
    rdd = sc.textFile(path)\
        .map(lambda line: line.split())\
        .map(lambda words: Row(label=words[0], words=words[1:]))
    return spark.createDataFrame(rdd)

# Load train and test data
train_data = load_dataframe(os.path.join(os.path.dirname(__file__), "phd-dataset/20ng-train-all-terms.txt")).persist()
test_data = load_dataframe(os.path.join(os.path.dirname(__file__), "phd-dataset/20ng-test-all-terms.txt")).persist()

# Learn the vocabulary of our training data
vectorizer = CountVectorizer(inputCol="words", outputCol="features")

# Convert string labels into integers
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")

# Learn a naive Bayes classifier
classifier = NaiveBayes(
    labelCol="label_index", featuresCol="features", predictionCol="label_index_predicted",
)

# Create pipeline with all above stages
pipeline = Pipeline(stages=[
    vectorizer,
    label_indexer,
    classifier
])

# Train model
model = pipeline.fit(train_data)

# Apply model to test data
predictions = model.transform(test_data)

# Evaluate prediction quality
evaluator = MulticlassClassificationEvaluator(labelCol="label_index",
                                              predictionCol="label_index_predicted",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = {:.2f}".format(accuracy))