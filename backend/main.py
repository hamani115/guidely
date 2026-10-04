from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Guidely backend is running"}