import subprocess
from pyspark.sql import SparkSession
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

subprocess.run(
    ["wget", "-O", "book.txt", "https://www.gutenberg.org/cache/epub/76938/pg76938.txt"],
    check=True
)

spark = SparkSession.builder.appName("NamedEntityCount").getOrCreate()
sc = spark.sparkContext

input = sc.textFile("book.txt")

lines = input.map(lambda s: s.strip())
nonempty = lines.filter(lambda s: s != "")

trees = nonempty.map(lambda s: ne_chunk(pos_tag(word_tokenize(s)), binary=False))
entities = trees.flatMap(
     lambda tree: [" ".join(tok for tok, _ in t.leaves())
                   for t in tree if isinstance(t, Tree)]
)

pairs = entities.map(lambda e: (e, 1))
counts = pairs.reduceByKey(lambda a, b: a + b)
swap = counts.map(lambda kv: (kv[1], kv[0]))
sort_swap = swap.sortBy(lambda x: (-x[0], x[1]))
sorted = sort_swap.map(lambda x: (x[1], x[0]))

results = sorted.collect()
for entity, count in results:
    print(entity, count)

spark.stop()
