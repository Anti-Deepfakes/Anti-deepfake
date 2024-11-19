from fastapi import APIRouter, HTTPException
import subprocess

trigger_router = APIRouter()

@trigger_router.get("/trigger")
async def run_training_script():
    try:
        result = subprocess.run(
            ['/bin/bash', '/home/ubuntu/model/run_model.sh'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Script error: {e.stderr}")

@trigger_router.get("/trigger1")
async def run_secondary_training_script():
    try:
        result = subprocess.run(
            ['/bin/bash', '/home/ubuntu/model/run_model1.sh'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Script error: {e.stderr}")
