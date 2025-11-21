# src/spark_jobs/feature_engineering_spark.py

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler


def extract_token_embeddings(df):
    """Transform Spark NLP embedding column into array<float>."""
    return df.withColumn(
        "token_embeddings",
        F.expr("transform(embeddings, x -> x.embeddings)")
    )


def explode_embeddings(df):
    """Explode token embeddings into individual rows."""
    return df.select(
        "*",
        F.explode("token_embeddings").alias("emb")
    )


def compute_embedding_means(df, dims=100, group_cols=None):
    """Compute mean embedding vector for each document."""

    # Extract each dimension
    for i in range(dims):
        df = df.withColumn(f"dim_{i}", F.col("emb")[i])

    # Default: group per document ID
    if group_cols is None:
        group_cols = ["doc_id", "label"]

    agg_exprs = [F.avg(f"dim_{i}").alias(f"dim_{i}") for i in range(dims)]

    return df.groupBy(*group_cols).agg(*agg_exprs)


def vectorize_mean_embeddings(df, dims=100):
    """Convert dim_0..dim_99 into MLlib DenseVector."""

    assembler = VectorAssembler(
        inputCols=[f"dim_{i}" for i in range(dims)],
        outputCol="features"
    )
    return assembler.transform(df)

