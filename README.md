# OpertusMundi Recommender System v0.0.1 

The Recommender Service provides personalized asset recommendations to the users of the OpertusMundi marketplace. 
This allows the user to discover a wide array of related geospatial assets and value-added services they might be interested in, 
considering their preceding feedback on other assets in the platform

# Setup

##### Install Requirements
Install project dependencies by running the following from project root directory:
```
pip install -r requirements.txt
```
##### Run Recommender Server
Run the Recommender Service by executing the following from project root directory:
```
uvicorn main:app --reload
```

## Documentation

Once the server is running, access the Recommender System API Documentation at http://localhost:8000/docs

## References

1. [Asset Core Metadata Schema](https://github.com/OpertusMundi/catalogue-service/blob/master/catalogueapi/resources/core.json)
2. [Basic Collaborative Filter](https://medium.com/@lope.ai/recommendation-systems-from-scratch-in-python-pytholabs-6946491e76c2)