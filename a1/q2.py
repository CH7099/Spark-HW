# CS6350.002 Big Data Management & Analytics
# Assignment 1-2: Movie Summaries Analysis with PySpark


# ── Standard Library
import os, sys, re, math, subprocess
# ── Third-party
from pyspark.sql import SparkSession
from collections import Counter
from nltk.corpus import stopwords
# ── (Optional) nltk 다운로드 핸들링은 아래 try/except에서

def main():
    # ╔══════════════════════════════╗
    # ║ 0) Spark & Resources Setup   ║
    # ╚══════════════════════════════╝
    spark = SparkSession.builder.appName("Movie Summaries Analysis").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    
    # ╔══════════════════════════════╗
    # ║ 1) Ingest & Preprocessing    ║
    # ╚══════════════════════════════╝
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        import nltk
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    bc_stop_words = sc.broadcast(stop_words)
    
    TOKEN_RE = re.compile(r"[a-z0-9]+")
    
    if not os.path.exists("MovieSummaries/plot_summaries.txt"):
        subprocess.run(["wget", "-q", "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"], check=True)
        subprocess.run(["tar", "-xvzf", "MovieSummaries.tar.gz"], check=True)
        
    text = sc.textFile("MovieSummaries/plot_summaries.txt")
    
    # preprocessing
    def remove_stopwords(pair):
        movie_id, plot = pair
        stop_words = bc_stop_words.value
        tokens = TOKEN_RE.findall(plot.lower())
        filtered = [w for w in tokens if w not in stop_words]
        return movie_id, ' '.join(filtered)
    
    rdd = text.map(lambda x: (x.split("\t")[0], x.split("\t")[1])) # (movieID, plot)
    rdd = rdd.map(remove_stopwords) # (movieID, cleaned_plot)
    
    # ╔══════════════════════════════╗
    # ║ 2) TF / DF / IDF / TF-IDF    ║
    # ╚══════════════════════════════╝
    
    # Term Frequency (TF)
    words_rdd = rdd.flatMap(lambda x: [(word.lower(), x[0]) for word in x[1].split()]) # output: (word, movieID)
    tf_rdd = words_rdd.map(lambda x: ((x),1)).reduceByKey(lambda a,b: a+b) # map: ((term, movieID), 1) -> reduce: ((term, movieID), tf_count:int)
    
    # Document Frequency (DF)
    df_rdd = words_rdd.distinct().map(lambda x: (x[0],1)).reduceByKey(lambda a,b: a+b) # distinct -> map: (term, 1) -> reduce: (term, df_count)
    
    # IDF
    N = rdd.keys().distinct().count() # number of documents
    idf_rdd = df_rdd.map(lambda x: (x[0], math.log(N/x[1]))) # (term, idf_value = log(N/df))

    
    # TF-IDF
    tf_idf_rdd = tf_rdd.map(lambda x: (x[0][0], (x[0][1], x[1])))
    tf_idf_rdd = tf_idf_rdd.join(idf_rdd)
    tf_idf_rdd = tf_idf_rdd.map(lambda x: ((x[0], x[1][0][0]), x[1][0][1] * x[1][1]))
    # input: ((term, movieID), tf) -> map: (term, (movieID, tf)) -> join: (term, ((movieID, tf), idf)) -> map: ((term, movieID), tf*idf)    
    
    # ╔══════════════════════════════╗
    # ║ 3) Query Processing & Output ║
    # ╚══════════════════════════════╝
    movie_rdd = sc.textFile("MovieSummaries/movie.metadata.tsv")
    titles_rdd = movie_rdd.map(lambda x: x.split("\t")).map(lambda x: (x[0], x[2])) # (movieID, title)
    
        
    queries = [q.strip() for q in open("queries.txt", encoding='utf-8') if q.strip()]
    
    doc_norms = (tf_idf_rdd
            .map(lambda kv: (kv[0][1], kv[1] * kv[1]))  # (movieID, tfidf^2)
            .reduceByKey(lambda a,b: a+b)               # (movieID, Σ_t tfidf(t,d)^2)
            .mapValues(lambda s: math.sqrt(s))          # (movieID, ||d||)
            .cache())

    for query in queries:
        # single term query: results_rdd → joined → top10
        if len(query.split()) == 1:
            # (movieID, tfidf)
            results_rdd = (tf_idf_rdd.filter(lambda x: x[0][0] == query.lower()).map(lambda x: (x[0][1], x[1])))
            joined = results_rdd.join(titles_rdd) # joined: (movieID, (tfidf, title))
            top10 = (joined.map(lambda x: (x[1][0], x[1][1])).sortBy(lambda x: -x[0]).take(10)) # (tfidf, title)
            
            print(f"\nTop 10 documents for the term '{query}':")
            for rank, (score, title) in enumerate(top10, 1):
                print(f"{rank:>2}. {title}  (score={score:.6f})")
            
        # multiple term query: partial → dot → (join doc_norms) → cosine → top → named
        else: 
            # preprocess query
            tokenized_query = TOKEN_RE.findall(query.lower())
            filtered_query = [word for word in tokenized_query if word not in stop_words]
            q_tf = Counter(filtered_query)
            
            q_vector = {}
            for term, tf in q_tf.items():
                idf = idf_rdd.lookup(term)
                if idf:
                    q_vector[term] = tf * idf[0]
            
            # q_norm = ||q||
            q_norm = math.sqrt(sum(value**2 for value in q_vector.values()))
            
            # cosine similarity = dot / (||q||*||d||)
            q_terms = set(q_vector.keys())
            if not q_terms:
                print("  No results.")
                continue
            bc_qvector  = sc.broadcast(q_vector)
            bc_qterms  = sc.broadcast(q_terms)
            
            # dot = Σ_t tfidf_doc(t,d) * tfidf_query(t)
            partial = (tf_idf_rdd.filter(lambda kv: kv[0][0] in bc_qterms.value).map(lambda kv: (kv[0][1], kv[1] * bc_qvector.value.get(kv[0][0], 0.0))))  # (docID, contrib)
            dot = partial.reduceByKey(lambda a,b: a+b)  # (docID, dot)

            
            # cosine = dot / (||d|| * ||q||)
            cosine = (dot
                    .join(doc_norms)                         # (docID, (dot, d_norm))
                    .mapValues(lambda v: v[0] / (v[1] * q_norm))
                    .filter(lambda kv: kv[1] > 0.0))
            top = cosine.top(10, key=lambda kv: kv[1])        # [(docID, score), ...]
            named = (sc.parallelize(top)
                    .join(titles_rdd)                          # (docID, (score, title))
                    .map(lambda x: (x[1][0], x[1][1]))         # (score, title)
                    .sortBy(lambda kv: -kv[0])
                    .collect())

            print(f"\nTop 10 documents for the query '{query}':")
            if not named:
                print("  No results.")
            else:
                for rank, (score, title) in enumerate(named, 1):
                    print(f"{rank:>2}. {title}  (score={score:.6f})")
    
    sc.stop()

if __name__ == "__main__":
    main()