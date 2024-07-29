import os
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from utils import MODELS_DIR, VECTORIZER


def train_model(df):
    if "SGDClassifier.pkl" not in os.listdir(f"{MODELS_DIR}"):
        x, y = (
            df.select("text_processed").rdd.flatMap(lambda x: x).collect(),
            df.select("sentiment_processed").rdd.flatMap(lambda x: x).collect(),
        )

        print("[ MODEL ] VECTORIZER")
        train_vectors = VECTORIZER.fit_transform(x)

        print("[ MODEL ] FIT SGDClassifier")
        clf = SGDClassifier()

        distributions = dict(
            loss=["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
            learning_rate=["optimal", "invscaling", "adaptive"],
            eta0=uniform(loc=1e-7, scale=1e-2),
        )

        random_search_cv = RandomizedSearchCV(
            estimator=clf, param_distributions=distributions, cv=5, n_iter=200
        )
        random_search_cv.fit(train_vectors, y)
        print(f"[ MODEL ] BEST PARAMS: {random_search_cv.best_params_}")
        print(f"[ MODEL ] BEST SCORE: {random_search_cv.best_score_}")

        model = random_search_cv.best_estimator_

        with open(f"{MODELS_DIR}/SGDClassifier.pkl", "wb") as fid:
            pickle.dump(model, fid)

        with open(f"{MODELS_DIR}/tfidf.pkl", "wb") as tfidf:
            pickle.dump(VECTORIZER, tfidf)

        print("[ MODEL ] COMPLETED")
