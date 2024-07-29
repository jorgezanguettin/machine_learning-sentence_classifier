import os
from unidecode import unidecode
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType
from utils import DATASETS_DIR, STOPWORDS, LEMMATIZER, TOKENIZER


def process_data():
    if "imdb-reviews.csv" not in os.listdir(f"{DATASETS_DIR}/processed/"):
        print("[ SPARK ] STARTING")
        spark = SparkSession.builder.appName("DataframeProcess").getOrCreate()

        df = spark.read.csv(
            f"{DATASETS_DIR}/raw/imdb-reviews-pt-br.csv", sep=";", header=True
        )
        df = df.na.drop()

        string_process = udf(lambda x: func_string_process(x), StringType())
        target_process = udf(lambda x: func_target_process(x), IntegerType())

        print("[ SPARK ] PROCESSING")
        df = df.withColumn("text_processed", string_process(col("text_pt")))
        df = df.withColumn("sentiment_processed", target_process(col("sentiment")))

        print("[ SPARK ] FILTERING")
        df_filter = df.select(["text_processed", "sentiment_processed"])

        save_csv_filesystem("imdb-reviews", df_filter)

        return df_filter


def save_csv_filesystem(filename, dataframe):
    dataframe.toPandas().to_csv(
        f"{DATASETS_DIR}/processed/{filename}.csv", sep=";", header=True, index=False
    )


def func_string_process(text):
    return lemmatization_text(tokenizer_text(remove_stopwords(text)))


def func_target_process(target_value):
    if target_value == "neg":
        return 0
    else:
        return 1


def lemmatization_text(text):
    text_lemma = " ".join(
        list(map(lambda x: unidecode(LEMMATIZER.lemmatize(x)), text.split(" ")))
    )
    return text_lemma


def tokenizer_text(text):
    return " ".join(TOKENIZER.tokenize(text))


def remove_stopwords(text):
    return tokenizer_text(
        " ".join(
            list(
                filter(lambda x: x is not None and x not in STOPWORDS, text.split(" "))
            )
        )
    )
