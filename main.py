import uvicorn
from src.config import config

if __name__ == "__main__":
    uvicorn.run("src:app", host="0.0.0.0", port=config["port"], reload=True) 