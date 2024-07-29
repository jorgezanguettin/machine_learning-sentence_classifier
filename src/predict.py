import pickle
from src.data_preparation import func_string_process
from utils import MODELS_DIR

classes = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

def make_predictions(requests: list):
    with open(f"{MODELS_DIR}/SGDClassifier.pkl", "rb") as fid:
        model = pickle.load(fid)

    with open(f"{MODELS_DIR}/tfidf.pkl", "rb") as tfidf:
        vectorizer = pickle.load(tfidf)

    requests = list(
        map(
            lambda request: func_string_process(request),
            requests
        )
    )

    requests_transform = vectorizer.transform(requests)
    predictions = model.predict(requests_transform)

    for request, predict in zip(requests, predictions):
        print(f"[ MODEL ] PREDICT - {request} -> {classes[predict]}")

    return predictions
