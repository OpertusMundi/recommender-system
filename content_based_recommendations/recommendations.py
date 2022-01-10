import ast
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


def similarity(iri, model, number_of_recommendations=3):
    if model == 'RotatE':
        path = "EmbeddingModels/results/resultsRotatE/"
    elif model == 'TransH':
        path = "EmbeddingModels/results/resultsTransH/"
    # IRI = extractIRI(dataset)   # remove this

    entity_ids = open(path + "entities_ids.txt", "r")
    contents1 = entity_ids.read()
    entity = ast.literal_eval(contents1)
    id = entity[iri]

    # ToDO : Softcode these values
    all_datasets_ids = [3392, 3403, 3407, 3405, 3388, 3390, 3386, 3512, 3510, 3508, 3176, 3172, 3171, 974]

    model = torch.load(path + 'trained_model.pkl')
    entity_embeddings = model.entity_representations[0]
    original = entity_embeddings(torch.as_tensor(id)).detach().numpy()
    d = dict.fromkeys(all_datasets_ids)
    for i in range(len(all_datasets_ids)):
        embdding = entity_embeddings(torch.as_tensor(all_datasets_ids[i])).detach().numpy()
        #print("Is embedding complex(real and imaginary) in nature?", np.iscomplexobj(embdding))  # -> False
        cos_sim = cosine_similarity(original.reshape(1, -1), embdding.reshape(1, -1))
        d[all_datasets_ids[i]] = cos_sim
    # print(d)
    recommended_ids = sorted(d, key=d.get, reverse=True)[:number_of_recommendations]
    # print(recommended_ids.dtpye)
    # print(recommended_ids)
    print(*recommended_ids)
    return recommended_ids


# def extractIRI(name):               # if dataset name is given instead of iri
#     str = ".csv"
#     file_name = name.__add__(str)
#     path = "/home/cjain/PycharmProjects/RS_2021/data/Datasets/"
#     file_path = path.__add__(file_name)
#     df = pd.read_csv(file_path)
#     IRI = df.iat[0, 0]
#     return IRI

if __name__ == "__main__":
    # # dataset = "Grundwasserkörper, NGP 2009, Österreich"
    # iri = 'https://data.inspire.gv.at/e76c1db4-69ee-4252-aa0e-c7a65cf069f9'
    similarity(iri= 'https://data.inspire.gv.at/e76c1db4-69ee-4252-aa0e-c7a65cf069f9', model='RotatE', number_of_recommendations=4)
