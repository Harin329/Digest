import re
import csv
import spacy
import numpy as np
from fastapi import FastAPI
from classes import InputItem

app = FastAPI()
nlp = spacy.load("./model")

@app.get("/")
async def health_check():
    return "Welcome to Digest!", 200

@app.get("/version")
async def version_check():
    return "Version: 2022.3.10.17.38", 200

@app.post("/")
async def read_item(textObj: InputItem):
    avg_ingredient_vector = np.zeros(300)
    avg_action_vector = np.zeros(300)
    with open('reference/ingredient_vector.npy', 'rb') as ingredientFile:
        avg_ingredient_vector = np.load(ingredientFile)

    with open('reference/action_vector.npy', 'rb') as actionFile:
        avg_action_vector = np.load(actionFile)
    
    stepArray = re.split("[;:.!,?\n]", textObj.text)
    stepArray = [x.strip() for x in stepArray]
    stepArray = list(filter(None, stepArray))

    results = []

    for step in stepArray:
        hasIngredient = False
        hasAction = False

        tokens = nlp(step)
        for token in tokens:
            # print(token.text, token.has_vector, token.vector_norm, token.is_oov)
            if (not token.has_vector):
                continue
            cosineIngredient = np.dot(avg_ingredient_vector,token.vector)/(np.linalg.norm(avg_ingredient_vector)*np.linalg.norm(token.vector))
            # print(token.text + " INGREDIENT: " + str(cosineIngredient)) ## DEBUG
            cosineAction = np.dot(avg_action_vector,token.vector)/(np.linalg.norm(avg_action_vector)*np.linalg.norm(token.vector))
            # print(token.text + " ACTION: " + str(cosineAction)) ## DEBUG
            if (cosineIngredient > 0.4 and abs(cosineIngredient - cosineAction) > 0.05):
                hasIngredient = True
            if (cosineAction > 0.4 and abs(cosineIngredient - cosineAction) > 0.05):
                hasAction = True

            if (hasIngredient and hasAction):
                results.append(step)
                break

    return results, 200

@app.get("/tokens")
async def getTokens():
    tokenString = ""
    with open('reference/food.csv', newline='') as csvfile:
        foodReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in foodReader:
            tokenString += row[2] + " "
    tokens = nlp(tokenString)
    vectors = []
    for token in tokens:
        # print(token.text, token.has_vector, token.vector_norm, token.is_oov)
        vectors.append(token.vector)

    arr = np.array(vectors)
    avg_ingredient_vector = np.mean(arr, axis=0)
    print(avg_ingredient_vector)

    with open('reference/ingredient_vector.npy', 'wb') as ingredientFile:
        np.save(ingredientFile, avg_ingredient_vector)

    ingredientFile.close()

    tokenString = ""
    with open('reference/action.csv', newline='') as csvfile:
        foodReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in foodReader:
            tokenString += row[0] + " "
    tokens = nlp(tokenString)
    vectors = []
    for token in tokens:
        # print(token.text, token.has_vector, token.vector_norm, token.is_oov)
        vectors.append(token.vector)

    arr = np.array(vectors)
    avg_action_vector = np.mean(arr, axis=0)
    print(avg_action_vector)

    with open('reference/action_vector.npy', 'wb') as actionFile:
        np.save(actionFile, avg_action_vector)

    actionFile.close()

    return [avg_ingredient_vector.shape[0], avg_action_vector.shape[0]], 200
