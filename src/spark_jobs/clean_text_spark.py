# clean_text_spark.py
from spark_jobs.nlp_pipeline_spark import build_cleaning_pipeline

def run_spark_cleaning(df):
    pipeline = build_cleaning_pipeline()
    model = pipeline.fit(df)
    return model.transform(df)
