from fastapi import FastAPI
from routers import predict

app = FastAPI(title="API de Detecção de Fraude")

# Aqui você conecta o arquivo predict.py na sua API principal
app.include_router(predict.router, prefix="/ml", tags=["ML"])


@app.get("/")
def root():
    return {"message": "API de MLOps no ar!"}
