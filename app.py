import os
import time
import json
import logging
from datetime import datetime, timezone
from fastapi import FastAPI
from routes import router


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

app = FastAPI(
    docs_url='/api/docs',
    openapi_url='/api/openapi.json'
)

START_TIME = time.time()

app.include_router(router)

# Health check endpoint
@app.get("/ping")
async def health_check():
    return {
        "status": True,
        "message": "Text Eval Service is working"
    }

# Deep health check endpoint
@app.get("/health")
async def deep_health():
    return {
        "status": "ok",
        "uptime": time.time() - START_TIME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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