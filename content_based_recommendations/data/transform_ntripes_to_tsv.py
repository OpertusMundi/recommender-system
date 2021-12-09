# Convert knowledge graph in NTRIPLES format to tsv file

import pandas as pd

df = pd.read_csv('kgraph_geodata.nt',
                 sep=r'\s+',
                 header=None,
                 names=None,
                 dtype=str,
                 usecols=[0, 1, 2])

df = df.astype(str)

df[[0, 1, 2]] = df[[0, 1, 2]].replace({'>': ''}, regex=True)
df[[0, 1, 2]] = df[[0, 1, 2]].replace({'<': ''}, regex=True)

print(df.shape)

df.to_csv('data/kg_triples.tsv', index=False, header=False, sep='\t')


