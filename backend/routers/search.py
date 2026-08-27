from fastapi import APIRouter
from pydantic import BaseModel
from backend.service.search import search_documents

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    limit: int


@router.post("/search")
def send_query(request: SearchRequest):
    return search_documents(request.query, request.limit)

