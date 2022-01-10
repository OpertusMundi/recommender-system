from fastapi import FastAPI, Query
from fastapi.logger import logger
from pydantic import BaseModel
from typing import List, Union, Dict
from recommender import Recommender

# TODO: put this somewhere accessible
# TODO: enter service descriptions
tags_metadata = [
    {
        "name": "Recommender System",
        "description": "Description",
    }
]

# TODO: enter descriptions
app = FastAPI(
    title="OpertusMundi (top.io) Recommender System",
    description="",
    version="0.0.1"
)

recommender = Recommender()


@app.on_event("startup")
async def startup_event():
    logger.info("Service started")


@app.get("/recommender/assets", tags=["Recommender System"])
async def recommend_popular_assets(n: int = 1):
    """
    **Description:** Get a list of N popular assets

    **Parameters:**
    - **n**: Number of assets to be recommended, e.g., __5__
    """
    result = recommender.recommend_popular_assets(number_of_recommendations=n)
    return {"asset_id": result}


@app.get("/recommender/{asset_id}", tags=["Recommender System"])
async def recommend_by_asset_id(asset_id: int = None, n: int = 1):
    """
    **Description:** Recommend Assets Given Asset ID

    **Parameters:**
    - **asset_id**: ID of asset, e.g., __29__
    - **n**: Number of assets to be recommended, e.g., __10__
    """
    result = recommender.recommend_by_asset_id(asset_id=asset_id, number_of_recommendations=n)
    return {"asset_id": result}


@app.get("/recommender/{user_id}", tags=["Recommender System"])
async def recommend_by_user_id(user_id: int = None, n: int = 1):
    """
    **Description:** Recommend Assets Given User ID

    **Parameters:**
    - **asset_id**: ID of user, e.g., __31__
    - **n**: Number of assets to be recommended, e.g., __5__
    """
    result = recommender.recommend_by_user_id(user_id=user_id, number_of_recommendations=n)
    return {"asset_id": result}


@app.get("/recommender/{datasets}", tags=["Recommender System"])
async def recommend_datasets_on_contents(n: int = 2):
    """
    **Description:** Get a list of top N similar datasets

    **Parameters:**
    - **n**: Number of datasets to be recommended, e.g., __5__
    """
    result = recommender.recommend_datasets_on_contents(number_of_recommendations=n)
    return {"dataset_ids": result}

