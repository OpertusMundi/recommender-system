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


# NUMERIC TRIPLES  -- Some "postCode"s

tf = TriplesFactory.from_path(path)

training, testing, validation = tf.split([.8, .1, .1])

print(training)
print(testing)
print(validation)

print("num_entities", training.num_entities, "\n", "num_relations", training.num_relations, "\n", "num_triples",
      training.num_triples)


resultsComplExLiteral = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='ComplExLiteral',
    training_kwargs=dict(num_epochs=100),
    random_seed=1235,
    device='cpu',
)

modelComplExLiteral = resultsComplExLiteral.model
print("Trained model", modelComplExLiteral)
