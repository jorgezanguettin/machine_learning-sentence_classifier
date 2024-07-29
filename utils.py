import os
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
nltk.download("wordnet")

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR = f"{ROOT_DIR}/data"
MODELS_DIR = f"{ROOT_DIR}/models"

STOPWORDS = stopwords.words("portuguese")
TOKENIZER = nltk.RegexpTokenizer(r"\w+")

LEMMATIZER = WordNetLemmatizer()
VECTORIZER = TfidfVectorizer()
