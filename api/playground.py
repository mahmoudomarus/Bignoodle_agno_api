from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/status")
async def status():
    """
    Check if the playground service is available
    """
    return {"playground": "available"} 