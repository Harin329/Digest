# Digest

Welcome to Digest. This is an exploration of the Word2Vec algorithm. The goal of this project is to create an API that accepts a block of recipe steps and returns a parsed list of steps. The API is being applied in the app [Umami](https://github.com/ubclaunchpad/umami). The process of the project involves downloading a recipe set (RecipeNLG), using Google's Word2Vec algorithm to create word vector mappings, importing mappings into spaCy, calculating average vectors for a set of ingredients and actions and finally using these metrics to determine if a given recipe step is valid from a block of text.

## Prerequisites
- python
- pip
- recipeNLG dataset
    - [RecipeNLG](https://github.com/Glorf/recipenlg)
- word2vec codebase
    - [Google](https://code.google.com/archive/p/word2vec/)
    - [Word2Vec](https://github.com/tmikolov/word2vec)

## Usage
1. Git clone latest from repo
2. Download recipeNLG dataset into `data/` folder and unzip into csv format
3. Run `python3 process.py` to process the csv into a corpus text
4. `make` word2vec algorithm and run the commands according to the instructions `./word2vec -train ../data/directions.txt -output vectors.txt -cbow 0 -size 300 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 24 -binary 0 -iter 15`
5. Use the generated .txt file to train the spaCy model with `python -m spacy init vectors en ./word2vec/vectors.txt ./model`
6. Load the model into main.py and run GET token
7. Deploy & Enjoy!
    - Run `zip -r Digest.zip . -x '.??*'` to zip the project
    - Upload zip to Azure with:
    ```
    curl -X POST \               
    -H 'Content-Type: application/zip' \
    -u '$DigestAPI' \
    -T Digest.zip \
    https://digestapi.scm.azurewebsites.net/api/zipdeploy
    ```
