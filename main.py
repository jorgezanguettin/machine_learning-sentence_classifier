from src.data_preparation import process_data
from src.train_model import train_model
from src.predict import make_predictions


df = process_data()

train_model(df)
make_predictions(
    ["Esta um belo dia hoje", "Conteudo ruim, nao gostei", "Festa sensacional", "Ruim"]
)
