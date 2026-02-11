from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from gemini_client import GeminiClient
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI(title="AI Manga Generator API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:16332",
        "https://ai2comic.sparky.qzz.io",
        "https://ai2comic-api.sparky.qzz.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = GeminiClient()

class StoryboardRequest(BaseModel):
    prompt: str
    reference_style: Optional[str] = ""
    aspect_ratio: Optional[str] = "16:9"

class PanelImageRequest(BaseModel):
    panel_prompt: str
    style_reference: Optional[str] = ""
    aspect_ratio: Optional[str] = "16:9"
    image_size: Optional[str] = "1K"
    initial_image_base64: Optional[str] = None
    style_reference_image_base64: Optional[str] = None
    panel_id: str # To identify the image in the frontend

class BatchImageRequest(BaseModel):
    panels: List[PanelImageRequest]

class ApiKeyRequest(BaseModel):
    api_key: str

@app.get("/config/status")
async def get_config_status():
    return {"is_configured": client.client is not None}

@app.post("/config/api-key")
async def set_api_key(request: ApiKeyRequest):
    success = client.update_api_key(request.api_key)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save API key")
    return {"status": "updated"}


@app.post("/generate/storyboard")
async def generate_storyboard(request: StoryboardRequest):
    try:
        storyboard = await client.generate_storyboard(
            prompt=request.prompt,
            reference_style=request.reference_style,
            aspect_ratio=request.aspect_ratio
        )
        return storyboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse
import json as json_module

class StreamStoryboardRequest(BaseModel):
    prompt: str
    reference_style: Optional[str] = ""
    aspect_ratio: Optional[str] = "16:9"

@app.post("/generate/storyboard/stream")
async def generate_storyboard_stream(request: StreamStoryboardRequest):
    if not client.client:
        raise HTTPException(status_code=400, detail="API Key not configured")
    
    async def event_generator():
        try:
            async for chunk in client.generate_storyboard_stream(
                prompt=request.prompt,
                reference_style=request.reference_style,
                aspect_ratio=request.aspect_ratio
            ):
                yield f"data: {json_module.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json_module.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

class BatchPageRequest(BaseModel):
    pages: List[dict] # Full page objects
    style_reference: Optional[str] = ""
    aspect_ratio: Optional[str] = "9:16"
    image_size: Optional[str] = "2K"
    style_reference_image_base64: Optional[str] = None
    reference_prompt: Optional[str] = None  # From first page for consistency
    additional_image_base64: Optional[str] = None

@app.post("/generate/pages/batch")
async def generate_pages_batch(request: BatchPageRequest):
    if not client.client:
        raise HTTPException(status_code=400, detail="API Key not configured")

    async def process_page(page):
        try:
            image_base64 = await client.generate_manga_page(
                page_data=page,
                style_reference=request.style_reference,
                aspect_ratio=request.aspect_ratio,
                image_size=request.image_size,
                style_reference_image_base64=request.style_reference_image_base64,
                reference_prompt=request.reference_prompt,
                additional_image_base64=request.additional_image_base64
            )
            return {"page_number": page.get('page_number'), "image": image_base64}
        except Exception as e:
            print(f"Error processing page {page.get('page_number')}: {e}")
            return {"page_number": page.get('page_number'), "image": None}

    tasks = [process_page(p) for p in request.pages]
    results = await asyncio.gather(*tasks)
    
    return {"results": results}

class StreamPageRequest(BaseModel):
    page: dict
    style_reference: Optional[str] = ""
    aspect_ratio: Optional[str] = "9:16"
    image_size: Optional[str] = "2K"
    style_reference_image_base64: Optional[str] = None
    reference_prompt: Optional[str] = None
    additional_image_base64: Optional[str] = None

@app.post("/generate/page/stream")
async def generate_page_stream(request: StreamPageRequest):
    if not client.client:
        raise HTTPException(status_code=400, detail="API Key not configured")
    
    async def event_generator():
        try:
            async for chunk in client.generate_manga_page_stream(
                page_data=request.page,
                style_reference=request.style_reference,
                aspect_ratio=request.aspect_ratio,
                image_size=request.image_size,
                style_reference_image_base64=request.style_reference_image_base64,
                reference_prompt=request.reference_prompt,
                additional_image_base64=request.additional_image_base64
            ):
                yield f"data: {json_module.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json_module.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/generate/image")
async def generate_image(request: PanelImageRequest):
    try:
        image_base64 = await client.generate_panel_image(
            panel_prompt=request.panel_prompt,
            style_reference=request.style_reference,
            aspect_ratio=request.aspect_ratio,
            image_size=request.image_size,
            initial_image_base64=request.initial_image_base64,
            style_reference_image_base64=request.style_reference_image_base64
        )
        if not image_base64:
             raise HTTPException(status_code=500, detail="Failed to generate image")
        return {"panel_id": request.panel_id, "image": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/images/batch")
async def generate_images_batch(request: BatchImageRequest):
    """
    Generates images for multiple panels in parallel.
    """
    async def generate_one(panel_req):
        image_base64 = await client.generate_panel_image(
            panel_prompt=panel_req.panel_prompt,
            style_reference=panel_req.style_reference,
            aspect_ratio=panel_req.aspect_ratio,
            image_size=panel_req.image_size,
            initial_image_base64=panel_req.initial_image_base64,
            style_reference_image_base64=panel_req.style_reference_image_base64
        )
        return {"panel_id": panel_req.panel_id, "image": image_base64}

    tasks = [generate_one(panel) for panel in request.panels]
    results = await asyncio.gather(*tasks)
    
    # Filter out failed generations (where image is None) if any, or handle errors individually
    # For now, we return what we got.
    return {"results": results}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=16331)
