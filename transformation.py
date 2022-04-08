#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:36:45 2022

@author: nacharya
"""

from rdflib import Graph, Literal, RDF, URIRef,Namespace
from rdflib.namespace import XSD,OWL,RDFS
import pandas as pd
import ast
import json

mundi_uri= 'http://www.gra.fo/schema/untitled-ekg#'
dcat_uri='http://www.w3.org/ns/dcat#'
terms_uri='http://purl.org/dc/terms/'
version_uri='https://w3c.github.io/dxwg/dcat/#Property:'


with open('assets_2.jsonl') as file:
    json_datasets = [line.rstrip() for line in file]

json_datasets=[json.loads(item) for item in json_datasets]
g = Graph()
g.parse("topio-dcat.ttl",format="turtle")

for json_dataset in json_datasets:
    for key in json_dataset["_source"].keys():
        if(key=="id"):
            g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),RDF.type,URIRef(dcat_uri+'Dataset')))
        elif(key=="properties"):
            for key_property in json_dataset["_source"]["properties"].keys():
                if(key_property=='suitable_for'):
                    for topic in json_dataset["_source"]["properties"]["suitable_for"]:
                        topic=topic.replace(" ","")
                        g.add((URIRef(mundi_uri+topic),RDF.type,URIRef(mundi_uri+'Topic')))
                        g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(mundi_uri+'suitable_for'),URIRef(mundi_uri+topic)))
                elif(key_property=='topic_category'):
                    for topic in json_dataset["_source"]["properties"]["topic_category"]:
                        topic=topic.replace(" ","")
                        g.add((URIRef(URIRef(mundi_uri+topic)),RDF.type,URIRef(mundi_uri+'Topic')))
                        g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(dcat_uri+'theme'),URIRef(mundi_uri+topic)))
                elif(key_property=='language'):
                    g.add((URIRef(mundi_uri+json_dataset["_source"]["properties"]["language"]),RDF.type,URIRef(mundi_uri+'Language')))
                    g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(terms_uri+'language'),URIRef(mundi_uri+json_dataset["_source"]["properties"]["language"])))
                elif(key_property=='keywords'):
                    for keyword_js in json_dataset["_source"]["properties"]["keywords"]:
                        keyword_js["keyword"]=keyword_js["keyword"].replace(" ","")
                        g.add((URIRef(mundi_uri+keyword_js["keyword"]),RDF.type,URIRef(mundi_uri+'Keyword')))
                        g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(dcat_uri+'keyword'),URIRef(mundi_uri+keyword_js["keyword"])))
                elif(key_property=='abstract'):
                        abstract_literal=Literal(json_dataset["_source"]["properties"]["abstract"],datatype=XSD.string)
                        g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(terms_uri+'description'),abstract_literal))
                elif(key_property=='creation_date'):
                        create_date_literal=Literal(json_dataset["_source"]["properties"]["creation_date"],datatype=XSD.dateTime)
                        g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(mundi_uri+'creation_date'),create_date_literal))
                elif(key_property=='resources'):
                    for resource in json_dataset["_source"]["properties"]["resources"]:
                        try:
                            format_literal=Literal(resource["format"],datatype=XSD.string)
                            g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(terms_uri+'format'),format_literal))
                        except:
                            print("No value for format")
                elif(key_property=='revision_date'):
                    try:
                        revision_date_date_literal=Literal(json_dataset["_source"]["properties"]["revision_date"],datatype=XSD.dateTime)
                        g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(mundi_uri+'revision_date'),revision_date_date_literal))
                    except:
                        print("No value for revision date")
                elif(key_property=='scales'):
                    for scale_js in json_dataset["_source"]["properties"]["scales"]:
                        scale_literal=Literal(scale_js["scale"],datatype=XSD.long)
                        g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(mundi_uri+'scale'),scale_literal))
                elif(key_property=='title'):
                    title_literal=Literal(json_dataset["_source"]["properties"]["title"],datatype=XSD.string)
                    g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(terms_uri+'title'),title_literal))
                elif(key_property=='versions'):
                    for version in json_dataset["_source"]["properties"]["versions"]:
                        version_literal=Literal(version,datatype=XSD.string)
                        g.add((URIRef(mundi_uri+json_dataset["_source"]["id"]),URIRef(version_uri+'resource_version'),version_literal))

                    
                        
                    
g.serialize('data/mundi_populated.ttl',format='ttl')                 
                        
g.parse("data/mundi_populated.ttl",format="turtle")
row_list=[]
covid_instance = 'http://speaker.fraunhofer.de/covid-19/instance/'
covid_vocab = 'http://speaker.fraunhofer.de/vocabs/covid-19#'
rdfs_str='http://www.w3.org/1999/02/22-rdf-syntax-ns#'

for s,p,o in g:
    dict_row={}
    dict_row["subject"]=s
    dict_row["predicate"]=p
    dict_row["object"]=o
    row_list.append(dict_row)

df = pd.DataFrame(row_list) 
df.to_csv('data.tsv', sep='\t',index=False) 
                        
                        
                    
                    
                    
                    
                    
            
    
