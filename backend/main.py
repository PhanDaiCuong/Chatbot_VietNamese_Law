from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch
import json
import logging
from typing import Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vietnamese Law Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = "CuongPhan23/Chatbot_VietNamese_Law_Qwen"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9

# Global model variables
tokenizer = None
model = None

def load_model():
    """Load model and tokenizer"""
    global tokenizer, model
    try:
        logger.info(f"Loading model with Unsloth: {MODEL_NAME}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_NAME,
            max_seq_length = 4096,
            dtype = None,
            load_in_4bit = True,
            device_map = "auto"
        )
        model = FastLanguageModel.for_inference(model)
        model.eval()
        logger.info("Model loaded successfully with Unsloth and prepared for inference")
    except Exception as e:
        logger.error(f"Failed to load model with Unsloth: {e}")
        raise

# Load model on startup
load_model()

class ChatRequest(BaseModel):
    user_message: str
    user_id: Optional[str] = "default"
    bot_id: Optional[str] = "vietnamese_law_bot"
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    content: str
    user_id: str
    bot_id: str

def format_prompt(message: str) -> str:
    """Format user message into proper prompt format"""
    return f"<|user|>\n{message}\n<|assistant|>\n"

def generate_response(prompt: str) -> str:
    """Generate response from model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        user_input = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        reply = generated_text.replace(user_input, "").strip()
        
        return reply
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")

def generate_streaming_response(prompt: str):
    """Generate streaming response (simulated)"""
    try:
        full_response = generate_response(prompt)
        
        # Simulate streaming by yielding chunks
        words = full_response.split()
        for i, word in enumerate(words):
            chunk = {
                "content": word + " ",
                "done": i == len(words) - 1
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.05)  # Small delay for streaming effect
            
        # Final chunk
        yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        error_chunk = {
            "error": "Failed to generate response",
            "done": True
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@app.post("/api/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Main chat endpoint with optional streaming"""
    try:
        logger.info(f"Received chat request from user: {request.user_id}")
        
        if not request.user_message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        
        prompt = format_prompt(request.user_message)
        
        if request.stream:
            return StreamingResponse(
                generate_streaming_response(prompt),
                media_type="text/plain"
            )
        else:
            response_content = generate_response(prompt)
            return ChatResponse(
                content=response_content,
                user_id=request.user_id,
                bot_id=request.bot_id
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": model is not None and tokenizer is not None
    }

# Legacy endpoints for backward compatibility
@app.post("/chat/complete")
async def legacy_chat_complete(request: ChatRequest):
    """Legacy endpoint - redirects to new API"""
    try:
        response_content = generate_response(format_prompt(request.user_message))
        return {
            "task_result": {
                "content": response_content
            }
        }
    except Exception as e:
        logger.error(f"Legacy endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
