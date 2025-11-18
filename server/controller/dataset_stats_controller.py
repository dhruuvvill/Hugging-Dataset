"""
Controller for dataset statistics endpoints.
"""

import json
import logging
from fastapi import APIRouter, HTTPException
from services.hf_service import HuggingFaceService
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()
hf_service = HuggingFaceService()


@router.get("/datasets/trending")
async def get_trending_datasets(limit: int = 10):
    """
    Get trending datasets from HuggingFace.
    
    Args:
        limit: Number of datasets to return (default: 10)
        
    Returns:
        List of trending datasets with their statistics
    """
    try:
        api = hf_service.api
        datasets = []
        
        # Fetch trending datasets (sorted by downloads, most recent)
        logger.info(f"Fetching top {limit} trending datasets...")
        
        # Get datasets sorted by downloads (descending)
        dataset_list = api.list_datasets(
            sort="downloads",
            direction=-1,
            limit=limit * 2  # Get more to filter
        )
        
        count = 0
        for dataset in dataset_list:
            if count >= limit:
                break
                
            try:
                dataset_info = {
                    "id": dataset.id,
                    "downloads": getattr(dataset, "downloads", 0) or 0,
                    "likes": getattr(dataset, "likes", 0) or 0,
                    "lastModified": str(getattr(dataset, "lastModified", "")) if hasattr(dataset, "lastModified") else None,
                    "tags": getattr(dataset, "tags", []) or [],
                    "author": dataset.id.split("/")[0] if "/" in dataset.id else "unknown",
                }
                
                # Calculate trending score (combination of downloads and recent activity)
                downloads = dataset_info["downloads"]
                likes = dataset_info["likes"]
                dataset_info["trending_score"] = downloads * 0.7 + likes * 100 * 0.3
                
                datasets.append(dataset_info)
                count += 1
            except Exception as e:
                logger.warning(f"Error processing dataset {dataset.id}: {e}")
                continue
        
        # Sort by trending score
        datasets.sort(key=lambda x: x["trending_score"], reverse=True)
        
        return {
            "datasets": datasets[:limit],
            "count": len(datasets[:limit]),
            "message": f"Top {limit} trending datasets"
        }
        
    except Exception as e:
        logger.error(f"Error fetching trending datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending datasets: {str(e)}")


@router.get("/datasets/top")
async def get_top_datasets(limit: int = 10, sort_by: str = "downloads"):
    """
    Get top datasets by downloads or likes.
    
    Args:
        limit: Number of datasets to return (default: 10)
        sort_by: Sort by 'downloads' or 'likes' (default: 'downloads')
        
    Returns:
        List of top datasets
    """
    try:
        api = hf_service.api
        
        # Validate sort_by parameter
        if sort_by not in ["downloads", "likes"]:
            raise HTTPException(status_code=400, detail="sort_by must be 'downloads' or 'likes'")
        
        logger.info(f"Fetching top {limit} datasets sorted by {sort_by}...")
        
        # Get datasets sorted by the specified field
        dataset_list = api.list_datasets(
            sort=sort_by,
            direction=-1,
            limit=limit * 2
        )
        
        datasets = []
        count = 0
        
        for dataset in dataset_list:
            if count >= limit:
                break
                
            try:
                dataset_info = {
                    "id": dataset.id,
                    "downloads": getattr(dataset, "downloads", 0) or 0,
                    "likes": getattr(dataset, "likes", 0) or 0,
                    "lastModified": str(getattr(dataset, "lastModified", "")) if hasattr(dataset, "lastModified") else None,
                    "tags": getattr(dataset, "tags", []) or [],
                    "author": dataset.id.split("/")[0] if "/" in dataset.id else "unknown",
                }
                
                datasets.append(dataset_info)
                count += 1
            except Exception as e:
                logger.warning(f"Error processing dataset {dataset.id}: {e}")
                continue
        
        return {
            "datasets": datasets,
            "count": len(datasets),
            "sort_by": sort_by,
            "message": f"Top {limit} datasets by {sort_by}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching top datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch top datasets: {str(e)}")


@router.get("/datasets/stats")
async def get_dataset_stats():
    """
    Get overall dataset statistics.
    
    Returns:
        Statistics about datasets on HuggingFace
    """
    try:
        api = hf_service.api
        
        logger.info("Fetching dataset statistics...")
        
        # Get a sample of datasets to calculate stats
        dataset_list = api.list_datasets(sort="downloads", direction=-1, limit=100)
        
        total_downloads = 0
        total_likes = 0
        dataset_count = 0
        tags_count = {}
        
        for dataset in dataset_list:
            try:
                downloads = getattr(dataset, "downloads", 0) or 0
                likes = getattr(dataset, "likes", 0) or 0
                tags = getattr(dataset, "tags", []) or []
                
                total_downloads += downloads
                total_likes += likes
                dataset_count += 1
                
                # Count popular tags
                for tag in tags[:5]:  # Top 5 tags per dataset
                    tags_count[tag] = tags_count.get(tag, 0) + 1
                    
            except Exception as e:
                continue
        
        # Get top tags
        top_tags = sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_datasets_sampled": dataset_count,
            "average_downloads": total_downloads / dataset_count if dataset_count > 0 else 0,
            "average_likes": total_likes / dataset_count if dataset_count > 0 else 0,
            "total_downloads": total_downloads,
            "total_likes": total_likes,
            "top_tags": [{"tag": tag, "count": count} for tag, count in top_tags],
            "message": "Dataset statistics"
        }
        
    except Exception as e:
        logger.error(f"Error fetching dataset stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dataset statistics: {str(e)}")

