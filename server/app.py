import traceback
from typing import Any, Dict, Optional
import os
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from env.environment import ContentModerationEnv
from env.models import Action

app = FastAPI(title="Content Moderation Environment OpenEnv Server")

# Global environment instance
current_env: Optional[ContentModerationEnv] = None

@app.post("/reset")
async def reset(request: Request):
    global current_env
    task_name = "classification"
    
    # Try to parse optional JSON body
    try:
        payload = await request.json()
        if isinstance(payload, dict) and "task_name" in payload:
            task_name = payload["task_name"]
    except Exception:
        pass # allow empty body
        
    try:
        current_env = ContentModerationEnv(task_name=task_name)
        obs = current_env.reset()
        return obs.model_dump()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
async def step(request: Request):
    global current_env
    if not current_env:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    
    # Support both flat JSON payload and nested {"action": ...} payload
    action_data = payload.get("action", payload) if isinstance(payload, dict) and "action" in payload else payload
    
    try:
        action = Action(**action_data)
        obs, reward, done, info = current_env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def state():
    global current_env
    if not current_env:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return current_env.state().model_dump()

@app.post("/state")
async def state_post():
    # Some environments post to state
    global current_env
    if not current_env:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return current_env.state().model_dump()

@app.get("/")
async def root():
    return {"status": "ok", "message": "OpenEnv Server Running. Use /reset and /step for inference."}

@app.head("/")
async def root_head():
    return JSONResponse(content={"status": "ok"})

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
