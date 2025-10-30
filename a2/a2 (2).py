
import os, re, subprocess, math
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer # type: ignore
from pyspark.ml.linalg import DenseVector # type: ignore
from pyspark.sql import SparkSession # type: ignore

VOCAB_SIZE = 0  # Total word: 13423
def preprocess_data(spark):
    
    subprocess.run(["wget", "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"])
    subprocess.run(["unzip", "sms+spam+collection.zip", "-d", "spam_data"])

    df = spark.read.csv("spam_data/SMSSpamCollection", sep="\t", inferSchema=True).toDF("label", "text")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    wordsData = tokenizer.transform(df)
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    stopWordRemoved = remover.transform(wordsData)
    dataset = stopWordRemoved.select("label", "filtered")
    
    return dataset

def text_to_vector(dataset):
    cv = CountVectorizer(inputCol="filtered", outputCol="features")
    cv_model = cv.fit(dataset)
    dataset = cv_model.transform(dataset).select("label", "features")
    global VOCAB_SIZE
    VOCAB_SIZE = cv_model.vocabulary.__len__()
    return dataset

def main():
    spark = SparkSession.builder.appName("naivebayes_classifier").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    sc = spark.sparkContext
    
    # 1. Data Preprocessing (tokenization, stemming, stop word removal)
    dataset = preprocess_data(spark)
    dataset = text_to_vector(dataset)       # raw data to feature vectors
    # (label)|(size, [indices], [values])
    
    # 2. Splitting the dataset into training and test sets
    (trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed=42)
    labelIndex = {"ham": 0, "spam": 1}
    trainingData = trainingData.rdd    # Row(label='ham', features=SparseVector(13423, {0: 2.0, 1: 1.0, ... 3819: 1.0}))
    trainingData = trainingData.map(lambda x : (labelIndex["ham"], x[1]) if x[0] == "ham" else (labelIndex["spam"], x[1])) # (0, SparseVector(...)) or (1, SparseVector(...))
    
    # 3. Training the Naive Bayes model
    trainingLabel = trainingData.map(lambda x: x[0])
    totalDataCount = trainingLabel.count()
    labelDataCount = trainingLabel.countByValue()   # {0: count_of_ham, 1: count_of_spam}
    probSpam = labelDataCount[1] / totalDataCount   # P(spam)
    probHam = 1 - probSpam                          # P(ham)
    print("# of total data:", totalDataCount)
    print("# of data per label (ham or spam):", labelDataCount)
    print("P(spam):", probSpam, "P(ham):", probHam)
    print("Vocabulary size:", VOCAB_SIZE)

    # P(w_i|C) to get P(textmessage|spam) and P(textmessage|ham) (aka likelihood)
    # laplace smoothing is applied on P(w_i|C) = (count(w_i, C) + 1) / (totalWordsInClass + vocabSize)
    # P(textmessage|spam) = P(w_1|spam) * P(w_2|spam) * ... * P(w_n|spam)
    conditionalProbabilities = {} # likelihood
    densedTrainingData = trainingData.map(lambda x: (x[0], DenseVector(x[1].toArray()))) # (label, DenseVector)
    labelWordCount = densedTrainingData.reduceByKey(lambda x, y: x + y).collectAsMap()
    for label, wordFreqVector in labelWordCount.items():
        wordCountPerClass = wordFreqVector.values.sum()
        probs = {}
        for i, count in enumerate(wordFreqVector):
            probs[i] = (count + 1) / (wordCountPerClass + VOCAB_SIZE)
            if i < 5:
                print(f"P(word {i} | class {label}):", probs[i]) # probWiGivenClass
        conditionalProbabilities[label] = probs
    print("----- likelihood calculated -----")
    
    # 4. Testing the model with test data
    testData = testData.rdd.map(lambda x : (labelIndex["ham"], x[1]) if x[0] == "ham" else (labelIndex["spam"], x[1]))
    densedTestData = testData.map(lambda x: (x[0], DenseVector(x[1].toArray()))).collect()
    answers = testData.map(lambda x: x[0]).collect()
    predictions = []
    for row in densedTestData:
        trueLabel, features = row
        logProbSpam = math.log(probSpam)
        logProbHam = math.log(probHam)
        for i, freq in enumerate(features):
            if freq > 0:
                logProbSpam = logProbSpam + freq * (math.log(conditionalProbabilities[1][i])) # 1: spam
                logProbHam = logProbHam + freq * (math.log(conditionalProbabilities[0][i])) # 0: ham
        predictions.append(1 if logProbSpam > logProbHam else 0)

    # 5. Evaluating the model (compare answers and predictions)
    correct = sum([1 for true, pred in zip(answers, predictions) if true == pred])
    total = len(answers)
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

    sc.stop()

if __name__ == "__main__":
    main()
