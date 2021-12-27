import os
import pickle
import sys

import torch
from matplotlib import pyplot as plt
import numpy as np
import pykeen
from pykeen.triples import TriplesFactory
from typing import List
from pykeen.pipeline import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

path = "../data/kg_triples.tsv"

tf = TriplesFactory.from_path(path)

training, testing, validation = tf.split([.8, .1, .1])

print(training)
print(testing)
print(validation)

print("num_entities", training.num_entities, "\n", "num_relations", training.num_relations, "\n", "num_triples",
      training.num_triples)

resultsTransH = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='TransH',
    training_kwargs=dict(num_epochs=40),
    random_seed=1235,
    device='cpu',
)

modelTransH = resultsTransH.model
print("Trained model", modelTransH)

# saving the model, results(losses, metrics, stopper, times) and metadata
save_location = 'recommender-system/contentbased_recommendersystem/results/resultsTransH/'  # this directory
resultsTransH.save_to_directory(save_location)
os.listdir(save_location)

# plots
resultsTransH.plot_losses()
plt.savefig('results/resultsTransH/TransH_losses.png', dpi=300)

with open(save_location + 'triples_factory.pkl', 'wb') as f:
    pickle.dump(tf, f)

# # load the trained model and the instance of TriplesFactory
# modelTransH = torch.load(save_location + 'trained_model.pkl')
# with open(save_location + 'triples_factory.pkl', 'rb') as f:
#     loaded_tfac = pickle.load(f)

# Embeddings
entity_id_t = torch.as_tensor(tf.entity_ids)
relation_id_t = torch.as_tensor(tf.relation_ids)

# saving the embeddings - ids to file
original_stdout = sys.stdout

with open(save_location + 'entities_ids.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print(tf.entity_id_to_label)
    sys.stdout = original_stdout  # Reset the standard output to its original value

f1 = open(save_location + 'entities_ids.txt', "r+")
input = f1.read()
input = input.replace(',', '\n')
f2 = open(save_location + 'entity_ids_sep.txt', "w+")
f2.write(input)
f1.close()
f2.close()

original_stdout1 = sys.stdout

with open(save_location + 'relation_ids.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print(tf.relation_to_id)
    sys.stdout = original_stdout1  # Reset the standard output to its original value

f1 = open(save_location + 'relation_ids.txt', "r+")
input = f1.read()
print(input)
input = input.replace(',', '\n')
print(input)
f2 = open(save_location + 'relation_ids_sep.txt', "w+")
f2.write(input)
f1.close()
f2.close()
# print("entity_id_t", entity_id_t)
# print("relation_id_t", relation_id_t)

entity_embeddings = modelTransH.entity_representations[0]
austria_embedding, denmark_embedding, france_embedding = entity_embeddings(
    torch.as_tensor([3392, 3407, 3390])).detach().numpy()

# print('Austria_embedding:', austria_embedding)
# print('Denmark_embedding:', denmark_embedding)
# print('France embedding:', france_embedding)

cos_sim1 = cosine_similarity(austria_embedding.reshape(1, -1), denmark_embedding.reshape(1, -1))
cos_sim2 = cosine_similarity(austria_embedding.reshape(1, -1), france_embedding.reshape(1, -1))
# print('Cosine similarity: %.3f,  %.3f' % cos_sim1, cos_sim2)

print("Cosine similarity between austria and denmark: %.3f", cos_sim1)
print("Cosine similarity between austria and france: %.3f", cos_sim2)


dst1 = distance.euclidean(austria_embedding,denmark_embedding)
dst2 = distance.euclidean(austria_embedding,france_embedding)
print("Euclidean distance: %.3f, %.3f", dst1, dst2)

similarity2 = 1 / (1 + dst2)
similarity1 = 1 / (1 + dst1)

print("Euclidean similarity between austria and denmark: %.3f", similarity1)
print("Euclidean similarity between austria and france: %.3f", similarity2)