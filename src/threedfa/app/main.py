from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.routes import images,faced

app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost"], # Cannot allow CORS wildcard and allow credentials
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(faced.router)
app.include_router(images.router)