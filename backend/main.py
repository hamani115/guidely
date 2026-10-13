from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Guidely API",
    version="0.1.0",
)


app.include_router(router)


@app.get("/")
def root():
    return {"message": "Guidely backend is running"}
