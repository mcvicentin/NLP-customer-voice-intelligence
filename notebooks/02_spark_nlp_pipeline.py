#!/usr/bin/env python
# coding: utf-8

# # This notebook was created in the **Google colab** environment
# 
# # Goals here:
# - Create a Big Data style pipeline with:
# 
#     - Large-scale text cleanliness
#     - Distributed tokenization
#     - Pre-trained embeddings[link text](https://) (words vectorization - _Word2Vec / GloVe / BERT-as-SentenceEmbeddings_)
#     - Efficient cluster processing
#     - Integrate with **MLlib** for fast classifiers

# In[2]:


# --------------------------------------------------
# Bloco 1 — Setup Spark (Google Colab)
# --------------------------------------------------

# se necessário, instale:
# Instalar Java
#!apt-get update -qq
#!apt-get install -y openjdk-11-jdk-headless > /dev/null

# Instalar PySpark e Spark NLP
#!pip install -q pyspark==3.5.0
#!pip install -q spark-nlp==5.4.0

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

print("Java OK")

from pyspark.sql import SparkSession
import sparknlp

spark = sparknlp.start()

print("Spark version:", spark.version)
print("Spark NLP version:", sparknlp.version())


# In[3]:


# upload files
from google.colab import files
uploaded = files.upload()


# In[4]:


# Read files content with spark
from pyspark.sql import functions as F

# path for the training file
train_path = "/content/train.ft.txt.bz2"

# read
df_raw = spark.read.text(train_path)

df_raw.show(5, truncate=False)
df_raw.printSchema()
print("Total rows:", df_raw.count())


# In[5]:


# split "value" column--created by spark--in __label__X and text
df = (
    df_raw
    .withColumn("label", F.regexp_extract("value", r"__label__(\d)", 1).cast("int"))
    .withColumn("text", F.regexp_replace("value", r"__label__\d\s*", ""))
    .drop("value")
)

df.show(5, truncate=False)
df.printSchema()

print("Total samples:", df.count())


# # Now, we will apply distributed cleanliness with Spark NLP
# 
# 1) Create Spark NLP pipeline
# 2) Normalize, remove undesired characters and upper-case
# 3) Tokenize
# 4) Remove stopwords
# 5) CHeck results
# 
# 

# In[6]:


import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

from pyspark.ml import Pipeline


# In[7]:


# pre-processing
# Converte texto bruto em "document"
document_assembler = (
    DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
)

# Tokenização
tokenizer = (
    Tokenizer()
        .setInputCols(["document"])
        .setOutputCol("token")
)

# Normalização (remover símbolos, passar para minúsculas)
normalizer = (
    Normalizer()
        .setInputCols(["token"])
        .setOutputCol("normalized")
        .setLowercase(True)
)

# Remover stopwords
stopwords_cleaner = (
    StopWordsCleaner()
        .setInputCols("normalized")
        .setOutputCol("cleanTokens")
        .setCaseSensitive(False)
)

# Converter o resultado para colunas Spark normais
finisher = (
    Finisher()
        .setInputCols(["cleanTokens"])
        .setOutputCols(["clean_tokens"])
        .setCleanAnnotations(True)
)

# pipeline de limpeza sem Finisher -> aplicaremos na pipeline B, ao inserir os embeddings
nlp_clean_pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    normalizer,
    stopwords_cleaner
])


# In[8]:


# test in a small sub-sample
df_small = df.limit(5)
clean_model = nlp_clean_pipeline.fit(df_small)
clean_df = clean_model.transform(df_small)

clean_df.select("cleanTokens").show(truncate=False)


# # Now, let's generate embeddings!

# In[9]:


# let's first operate with a pre-trained glove: faster
from sparknlp.annotator import WordEmbeddingsModel

glove_embeddings = (
    WordEmbeddingsModel.pretrained("glove_100d")
        .setInputCols(["document", "cleanTokens"])   # usa tokens limpos
        .setOutputCol("embeddings")
)


# In[10]:


embedding_pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    normalizer,
    stopwords_cleaner,
    glove_embeddings
])


# In[11]:


# run embedding pipeline
clean_model = nlp_clean_pipeline.fit(df_small)
clean_df = clean_model.transform(df_small)

emb_model = embedding_pipeline.fit(df_small)
emb_df = emb_model.transform(clean_df)


# In[12]:


# transform embeddings to vectorized float arrays
from pyspark.sql import functions as F

df_extracted = emb_df.withColumn(
    "token_embeddings",
    F.expr("transform(embeddings, x -> x.embeddings)")
)


# explode para linhas separadas
df_exploded = df_extracted.select(
    "label",
    F.explode("token_embeddings").alias("emb")
)

df_exploded.select("emb").show(5)


# In[13]:


# calculate mean for each array column 100dim
dims = 100  # glove_100d

for i in range(dims):
    df_exploded = df_exploded.withColumn(f"dim_{i}", F.col("emb")[i])


# In[14]:


# group by label e calcula média
agg_exprs = [F.avg(f"dim_{i}").alias(f"dim_{i}") for i in range(dims)]
df_avg = df_exploded.groupBy("label").agg(*agg_exprs)


# In[15]:


# mount final dense vector
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=[f"dim_{i}" for i in range(dims)],
    outputCol="features"
)

df_final = assembler.transform(df_avg)
df_final.select("label", "features").show(1, truncate=False)


# # Next, let the games begin:
# - Use whole sample -> run pipelines again
# - DIstributed classification with Spark MLlib
# - Let's train three large-scle classifier models:
#   - Logistic Regression (best baseline; see notebook 03_sentiment_modeling.ipynb)
#   - Random Forest (works well with dense vectors)
#   - Naive Bayes (Fast)

# In[23]:


df_with_id = df.withColumn("doc_id", F.monotonically_increasing_id())
df_all = df_with_id.limit(50000)

# run clean+embedding pipeline
clean_model_all = nlp_clean_pipeline.fit(df_all)
clean_df_all = clean_model_all.transform(df_all)

emb_model_all = embedding_pipeline.fit(df_all)
emb_df_all = emb_model_all.transform(clean_df_all)


# transform embeddings to vectorized float arrays
df_extracted_all = emb_df_all.withColumn(
    "token_embeddings",
    F.expr("transform(embeddings, x -> x.embeddings)")
).select("doc_id", "label", "token_embeddings")


# explode para linhas separadas
df_exploded_all = df_extracted_all.select(
    "doc_id",
    "label",
    F.explode("token_embeddings").alias("emb")
)

# calculate mean for each array column 100dim
dims = 100  # glove_100d

for i in range(dims):
    df_exploded_all = df_exploded_all.withColumn(f"dim_{i}", F.col("emb")[i])

# group by label e calcula média
agg_exprs = [F.avg(f"dim_{i}").alias(f"dim_{i}") for i in range(dims)]
df_avg_all = df_exploded_all.groupBy("doc_id", "label").agg(*agg_exprs)

# mount final dense vector
assembler = VectorAssembler(
    inputCols=[f"dim_{i}" for i in range(dims)],
    outputCol="features"
)

df_final_all = assembler.transform(df_avg_all)
#df_final_all.select("label", "features").show(1, truncate=False)


# In[24]:


# converting "label" for double
df_ml = df_final_all.select(
    F.col("label").cast("double").alias("label"),
    "features"
)

df_ml.show(10, truncate=False)


# In[25]:


# split train/test samples
train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)
train_df.count(), test_df.count()


# In[26]:


# Logistic reg
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.01
)

lr_model = lr.fit(train_df)
preds = lr_model.transform(test_df)

preds.select("label", "prediction", "probability").show(5, truncate=False)


# In[27]:


# evaluate
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(preds)
accuracy


# In[28]:


from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50
)

rf_model = rf.fit(train_df)
rf_preds = rf_model.transform(test_df)

rf_acc = evaluator.evaluate(rf_preds)
rf_acc


# --------------------
# Important observations
# --------------------
# 
# Before we move on, it is important to pinpoint some conclusions about why models trained in Spark NLP reached accuracy ~0.78, while simple models in scikit-learn ~0.89 (tested in notebook 03_sentiment_modeling.ipynb).
# 
# _________________________________________
# ### 1. Simple models (TF-IDF + Logistic Regression ou Linear SVM) performed better
# 
# IN a local environment (scikit-learn), we use:
# 
# - bag-of-words / TF-IDF
# - Regressão Logística
# - Linear SVM
# 
# These models reached acc ~ 0.89, well above embedding pipeline developed here.
# TF-IDF captures well discriminative features in short/medium texts, and these algorithms are very efficient in sparse spaces.
# _______________________________________
# 
# ### 2. Why Spark NLP (GloVe embeddings) had lower accuracy (~78%)?
# 
# In pipeline Spark, we use:
# 
# - static embeddings GloVe 100d
# - Average tokens embeddings
# - distributed logistic regression
# 
# This combination has several limitations:
# 
# - GloVe is static: no contexto, same vector for each independent word in the phrase-.
# - Averaging embeddings destroys completely the syntax structure and partially the meaning.
# - Linear distributed classifier do not compensate the semantic limitation.
# 
# 
# *Spark NLP + GloVe ~ 2014 models -> lower acc.
# ______________________________
# 
# ### 3. Spark is much slower in smaller datasets
# 
# Even with only 50k registers, logistic regression took ~25min in Google colab environment. This happens because Spark MLlib:
# 
# - create one local cluster
# - divide in partitions
# - execute distributed jobs
# - move data between JVM and Python
# - Only efficient for datasets > 5–10 GB
# - optimized for clusters, not individual notebooks
# _________________________________________
# 
# ### 5. Alternatives for better performances
# 
# - BERT embeddings via Spark NLP (requires GPU -> not available in Colab Free)
# - Fine-tuning BERT local + embeddings extraction + distributed classification
# - Use sentence-level models (Sentence-BERT) and load embeddings using Spark
# _______________________________________
# 
# -------------
# >  **Next, we will proceed with experiments using MLflow**
# ------------
# 
# - end-to-end MLflow experiment
# - Spark MLlib model (Logistic Regression)
# - metric log
# - parametres log
# - DataFrame schema log
# - Spark model log

# In[32]:


# initiate MLflow
import mlflow
import mlflow.spark


mlflow.set_experiment("spark_nlp_sentiment")
mlflow.set_tracking_uri("file:///content/mlruns")  # Colab


# In[33]:


from pyspark.ml.classification import LogisticRegression

with mlflow.start_run():

    # modelo
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=20,
        regParam=0.1
    )

    lr_model = lr.fit(train_df)

    # predições
    pred = lr_model.transform(test_df)

    # métricas
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    acc = evaluator.evaluate(pred)

    # log params
    mlflow.log_param("regParam", 0.1)
    mlflow.log_param("maxIter", 20)

    # log metrics
    mlflow.log_metric("accuracy", acc)

    # salvar modelo
    mlflow.spark.log_model(lr_model, "spark_lr_model")

    print("Accuracy:", acc)


# In[34]:


mlflow.search_runs()


# In[35]:





# In[ ]:




