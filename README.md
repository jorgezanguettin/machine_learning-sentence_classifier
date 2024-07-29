# Machine Learning - SGDClassifier for sentiments analysis.

## Table of Contents

- [About the Project](#about-the-project)
  - [Built With](#built-with)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Run Project ](#run-project)

## About the Project

**[PT]** Projeto desenvolvido com o propósito de utilizar a grande variedade de bibliotecas Python para realizar a construção de um algoritmo de Machine Learning para a classificação dos sentimentos de frases entre Positivo e Negativo.

**[EN]** Project developed with the purpose of using a wide variety of Python libraries to build a Machine Learning algorithm for classifying the sentiments of sentences between Positive and Negative.

### Built With

- [Python](https://www.python.org/)
    - [NLTK](https://www.nltk.org/)
    - [PySpark](https://spark.apache.org/docs/latest/api/python/index.html)
    - [Scikit Learn](https://scikit-learn.org/stable/)
    - [Scipy](https://scipy.org/)
    - [Unidecode](https://pypi.org/project/Unidecode/)
- [Dataset Kaggle](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr?resource=download)


### Prerequisites

- Python 3.10
- PIP (Python package manager)

### Installation

1. **[PT]** Clone o repositório: | **[EN]** Clone the repository:
    ```bash
    git clone git@github.com:jorgezanguettin/machine_learning-sentence_classifier.git
    ```
2. **[PT]** Navegue para o diretório do projeto: | **[EN]**  Navigate to the project directory:
    ```bash
    cd machine_learning-sentence_classifier
    ```
3. **[PT]** Crie um ambiente virtual: | **[EN]** Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. **[PT]** Instale as dependencias: | **[EN]** Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Run Project

**[PT]**

Para rodar o projeto, basta executar o arquivo **main.py**, que todo o processo de Machine Learning
será executado. Execute-o com o seguinte comando:
```bash
python main.py
```
Ao executar esse comando, os seguintes passos serão executadosÇ
1. Preparação dos dados - Utilizando PySpark, são realidos filtros e processamentos nos dados do
dataset
2. Treino do modelo - Utilizando Scikit Learn e Scipy, o modelo será treinado utilizando o conjunto
de dados processado
3. Predição de dados - Utilizando o modelo ja treinado e salvo, dados desconhecidos são inseridos
para o modelo prever entre as duas classes (Positivo e Negativo).

**[EN]**

To run the project, simply run the **main.py** file, which carries out the entire Machine Learning process
will be executed. Run it with the following command:
```bash
python main.py
```
When executing this command, the following steps will be performed
1. Data preparation - Using PySpark, real filters and processing of the data are performed
data set
2. Model training - Using Scikit Learn and Scipy, the model will be trained using the set
of processed data
3. Data prediction - Using the already trained and saved model, unknown data is entered
for the model to predict between the two classes (Positive and Negative).
