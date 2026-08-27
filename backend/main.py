from fastapi import FastAPI
from pydantic import BaseModel
from backend.routers.search import router as search_router
app = FastAPI()
app.include_router(search_router)

@app.get("/")
def hello():
    return {"message": "hello world"}


@app.get("/health")
def health():
    return {"message": "health"}
