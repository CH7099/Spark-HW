from pyspark.sql import SparkSession
import subprocess
from pyspark.sql import functions as F
from pyspark.sql.functions import split, col, explode, least, greatest
from pyspark.sql.window import Window
spark = (
    SparkSession.builder
    .appName("MutualFriends")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)
subprocess.run(
    ["wget", "https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt"], check=True
)
df = spark.read.text("soc-LiveJournal1Adj.txt")
df.show()
dfClean = df.filter(col("value").contains("\t"))
df = (dfClean.withColumn("user", split(col("value"), "\\t").getItem(0)).withColumn("friends", split(col("value"), "\\t").getItem(1)). withColumn("friend", split(col("friends"), ",")).select("user", "friend"))
df.show()

dfExploded = df.withColumn("friendSigle", explode(col("friend")))

targets = (
    dfExploded.select(col("user").cast("int").alias("user")).distinct().orderBy(F.rand()).limit(10))

dfTargets = dfExploded.join(targets, on="user", how="inner")

dfPair = dfTargets.alias("a").join(dfTargets.alias("b"), on="user") \
    .where(col("a.friendSigle") < col("b.friendSigle")) \
    .select(col("user").alias("mutualFriend"),
            col("a.friendSigle").alias("f1"),
            col("b.friendSigle").alias("f2"))


dfDirect = dfTargets.select(col("user").alias("df1"), col("friendSigle").alias("df2"))


dfDirectNorm = dfDirect.select(least(col("df1").cast("int"), col("df2").cast("int")).alias("f1"), greatest(col("df1").cast("int"), col("df2").cast("int")).alias("f2")).distinct()

dfDirectNorm.filter(col("f1") > col("f2")).count()
dfPair.filter(col("f1") > col("f2")).count()

dfNoDirect = dfPair.join(dfDirectNorm, on=["f1", "f2"], how="left_anti")

dfMutualCount = (dfNoDirect.select("f1","f2","mutualFriend").distinct().groupBy("f1","f2").agg(F.countDistinct("mutualFriend").alias("mutualCount")))

# directed recommendation
a = dfMutualCount.select(F.col("f1").alias("src"), F.col("f2").alias("dst"), "mutualCount")
b = dfMutualCount.select(F.col("f2").alias("src"), F.col("f1").alias("dst"), "mutualCount")
directed = a.unionByName(b)

w = Window.partitionBy("src").orderBy(F.col("mutualCount").desc(), F.col("dst").asc())
top10 = directed.withColumn("rk", F.row_number().over(w)).where(F.col("rk") <= 10)

# format
out = (
    top10.groupBy("src")
    .agg(F.collect_list(F.struct("rk", "dst")).alias("lst"))
    .select(
        "src",
        F.expr("concat_ws(',', transform(array_sort(lst), x -> cast(x.dst as string)))").alias("top10")
    )
)

sample10 = out.orderBy(F.rand()).limit(10)
sample10.show(truncate=False)
lines = sample10.selectExpr("concat(cast(src as string), '\t', top10) as line")
with open("q1_output.txt", "w", encoding="utf-8") as f:
    for row in lines.toLocalIterator():
        f.write(row["line"] + "\n")

spark.stop()