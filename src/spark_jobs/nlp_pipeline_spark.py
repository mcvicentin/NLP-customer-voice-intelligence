# src/spark_jobs/nlp_pipeline_spark.py

from sparknlp.annotator import WordEmbeddingsModel,Tokenizer, Normalizer, StopWordsCleaner
from pyspark.ml import Pipeline
from .clean_text_spark import build_cleaning_pipeline



def build_cleaning_pipeline(finish_output=False):
    """Builds a Spark NLP cleaning pipeline.
    
    Parameters
    ----------
    finish_output : bool
        If True, return also the Finisher to produce clean Python lists.
    """

    document = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized") \
        .setLowercase(True)

    stopwords_cleaner = StopWordsCleaner() \
        .setInputCols("normalized") \
        .setOutputCol("cleanTokens") \
        .setCaseSensitive(False)

    if finish_output:
        finisher = Finisher() \
            .setInputCols(["cleanTokens"]) \
            .setOutputCols(["clean_tokens"]) \
            .setCleanAnnotations(True)

        pipeline = Pipeline(stages=[
            document,
            tokenizer,
            normalizer,
            stopwords_cleaner,
            finisher
        ])
    else:
        pipeline = Pipeline(stages=[
            document,
            tokenizer,
            normalizer,
            stopwords_cleaner
        ])

    return pipeline



def build_embedding_pipeline(embedding_name="glove_100d"):
    """Creates a Spark NLP pipeline that:
    - cleans text
    - tokenizes
    - applies static embeddings
    
    Parameters
    ----------
    embedding_name : str
        Name of the pretrained Spark NLP embedding model.
    """

    cleaning_pipeline = build_cleaning_pipeline(finish_output=False)

    glove = WordEmbeddingsModel.pretrained(embedding_name) \
        .setInputCols(["document", "cleanTokens"]) \
        .setOutputCol("embeddings")

    pipeline = Pipeline(stages=[
        *cleaning_pipeline.getStages(),
        glove
    ])

    return pipeline
