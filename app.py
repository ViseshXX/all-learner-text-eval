import os
from fastapi import FastAPI
from routes import router

app = FastAPI(
    docs_url='/api/docs',
    openapi_url='/api/openapi.json'
)

app.include_router(router)

# Health check endpoint
@app.get("/ping")
async def health_check():
    return {
        "status": True,
        "message": "Text Eval Service is working"
    }

if __name__ == "__main__":
    import uvicorn
    num_workers = os.cpu_count() or 1
   
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=5001, 
        workers=num_workers,
        timeout_keep_alive=300,  # 5 min for large audio file
        limit_concurrency=1000
    )