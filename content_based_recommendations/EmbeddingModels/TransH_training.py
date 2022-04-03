import os
import pickle
import sys

import torch
from matplotlib import pyplot as plt
import numpy as np
import pykeen
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# path = "../data/kg_triples.tsv"
path = "../data_official/country.tsv"
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
# save_location = 'recommender-system/contentbased_recommendersystem/results/resultsTransH/'  # this directory
save_location = 'results_official/resultsTransH/'  # this directory
resultsTransH.save_to_directory(save_location)
os.listdir(save_location)

# plots
resultsTransH.plot_losses()
plt.savefig('results_official/resultsTransH/transH_losses.png', dpi=300)

with open(save_location + 'triples_factory.pkl', 'wb') as f:
    pickle.dump(tf, f)

# # load the trained model and the instance of TriplesFactory
# modelTransH = torch.load(save_location + 'trained_model.pkl')
# with open(save_location + 'triples_factory.pkl', 'rb') as f:
#     loaded_tfac = pickle.load(f)


# Embeddings
entity_id_t = torch.as_tensor(tf.entity_ids)
relation_id_t = torch.as_tensor(tf.relation_ids)

original_stdout = sys.stdout

with open(save_location + 'entities_ids.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print(tf.entity_to_id)
    sys.stdout = original_stdout  # Reset the standard output to its original value


original_stdout1 = sys.stdout

with open(save_location + 'relation_ids.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print(tf.relation_to_id)
    sys.stdout = original_stdout1  # Reset the standard output to its original value

