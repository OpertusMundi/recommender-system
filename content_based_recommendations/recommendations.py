import ast
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


def similarity(dataset, number_of_recommendations=3):
    IRI = extractIRI(dataset)
    id = extractIDs(IRI)
    all_datasets_ids = [3392, 3403, 3407, 3405, 3388, 3390, 3386, 3512, 3510, 3508, 3176, 3172, 3171, 974]
    modelRotatE = torch.load('EmbeddingModels/results/resultsRotatE/trained_model.pkl')
    entity_embeddings = modelRotatE.entity_representations[0]
    original = entity_embeddings(torch.as_tensor(id)).detach().numpy()
    d = dict.fromkeys(all_datasets_ids)
    for i in range(len(all_datasets_ids)):
        embdding = entity_embeddings(torch.as_tensor(all_datasets_ids[i])).detach().numpy()
        print("Is embedding complex(real and imaginary) in nature?", np.iscomplexobj(embdding))  # -> False
        cos_sim = cosine_similarity(original.reshape(1, -1), embdding.reshape(1, -1))
        d[all_datasets_ids[i]] = cos_sim
    # print(d)
    recommended_ids= sorted(d, key=d.get, reverse=True)[:number_of_recommendations]
    print(recommended_ids)
    return recommended_ids


def extractIRI(name):
    str = ".csv"
    file_name = name.__add__(str)
    path = "/home/cjain/PycharmProjects/RS_2021/data/Datasets/"
    file_path = path.__add__(file_name)
    df = pd.read_csv(file_path)
    IRI = df.iat[0, 0]
    return IRI


def extractIDs(IRI):
    entity_ids = open("EmbeddingModels/results/resultsRotatE/entities_ids.txt", "r")
    contents1 = entity_ids.read()
    entity = ast.literal_eval(contents1)
    return entity[IRI]


if __name__ == "__main__":
    dataset = "Grundwasserkörper, NGP 2009, Österreich"
    similarity(dataset, 4)
