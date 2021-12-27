import os
import pickle
import sys

import torch
from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np
import pykeen
from pykeen.triples import TriplesFactory
from typing import List
from pykeen.pipeline import pipeline
from sklearn.metrics.pairwise import cosine_similarity

path = "../data/kg_triples.tsv"

tf = TriplesFactory.from_path(path)

training, testing, validation = tf.split([.8, .1, .1])

print(training)
print(testing)
print(validation)

print("num_entities", training.num_entities, "\n", "num_relations", training.num_relations, "\n", "num_triples",
      training.num_triples)

resultsRotatE = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='RotatE',
    training_kwargs=dict(num_epochs=100),
    random_seed=1235,
    device='cpu',
)

modelRotatE = resultsRotatE.model
print("Trained model", modelRotatE)

# # saving the model, results(losses, metrics, stopper, times) and metadata
save_location = 'results/resultsRotatE/'  # this directory
resultsRotatE.save_to_directory(save_location)
os.listdir(save_location)

# load the trained model and the instance of TriplesFactory
# modelRotatE = torch.load(save_location + 'trained_model.pkl')
#
#
# with open(save_location + 'triples_factory.pkl', 'rb') as f:
#     loaded_tfac = pickle.load(f)

#plots
resultsRotatE.plot_losses()
plt.savefig('results/resultsRotatE/rotatE_losses.png', dpi=300)

with open(save_location + 'triples_factory.pkl', 'wb') as f:
    pickle.dump(tf, f)
#
#
# # Embeddings
# entity_id_t = torch.as_tensor(tf.entity_ids)
# relation_id_t = torch.as_tensor(tf.relation_ids)
#
# original_stdout = sys.stdout
#
# with open(save_location + 'entities_ids.txt', 'w') as f:
#     sys.stdout = f  # Change the standard output to the file we created.
#     print(tf.entity_to_id)
#     sys.stdout = original_stdout  # Reset the standard output to its original value
#
#
# original_stdout1 = sys.stdout
#
# with open(save_location + 'relation_ids.txt', 'w') as f:
#     sys.stdout = f  # Change the standard output to the file we created.
#     print(tf.relation_to_id)
#     sys.stdout = original_stdout1  # Reset the standard output to its original value

