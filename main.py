import os
import sys
import nltk
import pickle
import shutil
import pandas as pd
from time import sleep
from unidecode import unidecode

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from scipy.stats import uniform

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.functions import col, udf

# Dataset by https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr?resource=download

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = f"{ROOT_DIR}/datasets"
MODELS_DIR = f"{ROOT_DIR}/models"


STOPWORDS = stopwords.words("portuguese")
TOKENIZER = nltk.RegexpTokenizer(r"\w+")
LEMMATIZER = WordNetLemmatizer()
VECTORIZER = TfidfVectorizer()


class SparkPipeline:
    def start_process(self):
        print("[ SPARK ] STARTING")
        spark = SparkSession.builder.appName("DataframeProcess").getOrCreate()

        print("[ SPARK ] READING")
        df = spark.read.csv(f"{DATASETS_DIR}/imdb-reviews-pt-br.csv", sep=",", header=True)

        string_process = udf(lambda x: self.string_process(x), StringType())
        target_process = udf(lambda x: self.target_process(x), IntegerType())

        print("[ SPARK ] PROCESSING")
        df = df.withColumn("text_processed", string_process(col("text_pt")))
        df = df.withColumn("sentiment_processed", target_process(col("sentiment")))

        df_filter = df.select(["text_processed", "sentiment_processed"])
        df_filter.show()

        print("[ SPARK ] SAVING")
        self.save_csv_filesystem(
            "imdb-reviews",
            df_filter
        )

    def string_process(self, text):
        return self.lemmatization_text(
            self.tokenizer_text(
                self.remove_stopwords(
                    text=text
                )
            )
        )

    @staticmethod
    def target_process(target_value):
        if target_value == "neg":
            return 0
        else:
            return 1

    def save_csv_filesystem(self, filename, dataframe):
        dataframe.repartition(1).write.csv(
            f"{DATASETS_DIR}/{filename}_folder",
            sep=";",
            header=True,
        )
        sleep(1)
        self.parse_csv_finished(
            filename
        )

    @staticmethod
    def parse_csv_finished(filename):
        csv_raw = None
        while True:
            print("[ SPARK ] WAIT TO FINISH")
            for file in os.listdir(f"{DATASETS_DIR}/{filename}_folder"):
                if ".crc" not in file and file != "_SUCESS":
                    csv_raw = file
                    break
                else:
                    csv_raw = None
            if csv_raw:
                break

        shutil.move(
            f"{DATASETS_DIR}/{filename}_folder/{csv_raw}",
            f"{DATASETS_DIR}/{filename}.csv",
        )
        sleep(2)
        shutil.rmtree(f"{DATASETS_DIR}/{filename}_folder")

    @staticmethod
    def lemmatization_text(text):
        text_lemma = " ".join(
            list(
                map(
                    lambda x: unidecode(LEMMATIZER.lemmatize(x)),
                    text.split(" ")
                )
            )
        )
        return text_lemma

    @staticmethod
    def tokenizer_text(text):
        return " ".join(TOKENIZER.tokenize(text))

    def remove_stopwords(self, text):
        return self.tokenizer_text(" ".join( 
            list(
                filter(
                    lambda x: x not in STOPWORDS,
                    text.split(" ")
                )
            )
        ))


class MLClassifier:
    def __init__(self):
        global VECTORIZER

        if "imdb-reviews.csv" not in os.listdir(DATASETS_DIR):
            SparkPipeline().start_process()

        if "SGDClassifier.pkl" not in os.listdir(MODELS_DIR):
            df = pd.read_csv(f"{DATASETS_DIR}/imdb-reviews.csv", sep=";")
            VECTORIZER.fit_transform(df.text_processed.values)

            model, vector = self.machine_learning_constructor(
                df.text_processed.values,
                df.sentiment_processed.values
            )

            with open(f"{MODELS_DIR}/SGDClassifier.pkl", "wb") as fid:
                pickle.dump(model, fid)

                pickle.dump(VECTORIZER, open(f"{MODELS_DIR}/tfidf.pkl", "wb"))

        with open(f"{MODELS_DIR}/SGDClassifier.pkl", "rb") as fid:
            model = pickle.load(fid)

        with open(f"{MODELS_DIR}/tfidf.pkl", "rb") as tfidf:
            VECTORIZER = pickle.load(tfidf)

        print("[ MODEL ] COMPLETED")
        self.machine_learning_predictor(model, string="Isto esta horrivel!")

    @staticmethod
    def machine_learning_predictor(model, string):
        string = SparkPipeline().string_process(string)

        predict_transform = VECTORIZER.transform([string])
        predict = model.predict(predict_transform)

        print("[ MODEL ] PREDICTED", "NEGATIVE" if not predict[0] else "POSITIVE")

    @staticmethod
    def machine_learning_constructor(x, y):
        print("[ MODEL ] VECTORIZER")
        train_vectors = VECTORIZER.fit_transform(x)

        print("[ MODEL ] FIT SGDClassifier")
        clf = SGDClassifier()

        distributions = dict(
            loss=["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
            learning_rate=["optimal", "invscaling", "adaptive"],
            eta0=uniform(loc=1e-7, scale=1e-2)
        )

        random_search_cv = RandomizedSearchCV(
            estimator=clf,
            param_distributions=distributions,
            cv=5,
            n_iter=200
        )
        random_search_cv.fit(train_vectors, y)
        print(f"[ MODEL ] BEST PARAMS: {random_search_cv.best_params_}")
        print(f"[ MODEL ] BEST SCORE: {random_search_cv.best_score_}")

        return random_search_cv.best_estimator_, train_vectors


MLClassifier()
