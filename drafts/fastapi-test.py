from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define a request model
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

# GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# GET with a parameter
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "query": q}

# POST endpoint
@app.post("/items/")
def create_item(item: Item):
    total_price = item.price + (item.tax or 0)
    return {"item": item, "total_price": total_price}
