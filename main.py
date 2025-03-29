from fastapi import FastAPI
from app.api import stock, submit
from app.core.config import templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from fastapi.requests import Request

# Define the lifespan function first
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.db import init_db
    init_db()
    yield

# Initialize the FastAPI app with the lifespan handler
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

# Include routers for modular endpoints
app.include_router(stock.router)
app.include_router(submit.router)

# Root endpoint for the homepage
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
