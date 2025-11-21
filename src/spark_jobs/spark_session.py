# spark_session.py
from pyspark.sql import functions as F
from spark_jobs.nlp_pipeline_spark import build_embedding_pipeline
from spark_jobs.feature_engineering_spark import (
    extract_token_embeddings,
    explode_embeddings,
    compute_embedding_means,
    vectorize_mean_embeddings
)

def run_full_embedding_pipeline(df, dims=100):
    pipeline = build_embedding_pipeline("glove_100d")

    model = pipeline.fit(df)
    embedded = model.transform(df)

    df_tok = extract_token_embeddings(embedded)
    df_exp = explode_embeddings(df_tok)

    df_avg = compute_embedding_means(df_exp, dims=dims, group_cols=["doc_id", "label"])
    df_final = vectorize_mean_embeddings(df_avg, dims=dims)

    return df_final
