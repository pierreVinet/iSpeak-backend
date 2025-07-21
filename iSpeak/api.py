# app/api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import asyncio
import json
import uuid
import os
from .pipeline import process_audio_orchestrator
from .utils import parse_segments

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ispeaktele.netlify.app",
        "https://therapy-science.ch",
        "https://ispeak.therapy-science.ch"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for tracking processing status
processing_status = {}

@app.post("/analyze")
async def analyze_audio(metadata: str = Form(None), upload: UploadFile = File(...), segments: str = Form(None)):
    print("metadata", metadata)
    
    # Parse metadata JSON string into dictionary
    metadata_dict = json.loads(metadata) if metadata else {}
    
    if not metadata_dict.get("user_id"):
        raise HTTPException(status_code=400, detail="user_id is required")
    
    if not metadata_dict.get("patient_id"):
        raise HTTPException(status_code=400, detail="patient_id is required")

    user_id = metadata_dict.get("user_id")

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Parse segments if provided
    parsed_segments = None
    if segments:
        # Try to parse with validation first
        parsed_segments = parse_segments(segments)
    else:
        print("No segments provided in request")
    
    # Initialize status for uploading
    processing_status[job_id] = {"status": "uploading", "filename": upload.filename, "message": "Uploading file..."}
    
    # Save uploaded file to user-specific directory
    upload_dir = Path("data") / "uploads" / user_id / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / upload.filename
    with dest.open("wb") as f:
        f.write(await upload.read())
    
    # Update status: Upload completed
    processing_status[job_id].update({
        "status": "uploading_completed", 
        "filename": dest.name,
        "message": "File upload completed successfully"
    })

    # Start processing with segments, passing the user_id
    asyncio.create_task(process_audio_orchestrator(job_id, dest, processing_status, parsed_segments, metadata_dict))
    
    return {"job_id": job_id, "status": "uploading_completed", "filename": dest.name}

@app.get("/status/{job_id}")
async def get_status_stream(job_id: str):
    """Server-Sent Events endpoint for real-time status updates"""
    
    # If job still doesn't exist after timeout, return 404
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found")
    
    async def event_stream():            
        while True:
            status = processing_status.get(job_id, {"status": "not_found"})
            
            # Send status update
            yield f"data: {json.dumps(status)}\n\n"
            
            # Close connection when done
            if status.get("status") in ["completed", "error"]:
                break
                
            await asyncio.sleep(0.5) 
    
    return StreamingResponse(
        event_stream(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/results/{user_id}/{job_id}")
async def get_results(user_id: str, job_id: str):
    """Retrieve the analysis results for a given user and job."""
    results_path = Path("data") / "uploads" / user_id / job_id / "results.json"
    
    if not results_path.exists():
        return JSONResponse(status_code=404, content={"message": "Results not found."})
        
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
        
    return results

@app.get("/assessments/{user_id}")
async def get_user_assessments(user_id: str):
    """Retrieve all completed assessments for a given user."""
    user_uploads_dir = Path("data") / "uploads" / user_id
    
    if not user_uploads_dir.exists():
        return []
    
    assessments = []
    
    # Scan through each job directory
    for job_dir in user_uploads_dir.iterdir():
        if job_dir.is_dir():
            results_file = job_dir / "results.json"
            if results_file.exists():
                try:
                    with results_file.open("r", encoding="utf-8") as f:
                        results = json.load(f)
                    
                    assessments.append(results)
                    
                except (json.JSONDecodeError, Exception) as e:
                    # Skip corrupted or incomplete files
                    print(f"Error reading results file {results_file}: {e}")
                    continue
    
    # Sort assessments by date (newest first)
    assessments.sort(key=lambda x: x.get("metadata", {}).get("date", ""), reverse=True)
    
    return assessments

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "iSpeak backend is running"}


