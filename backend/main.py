from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import uuid

app = FastAPI()

# Load model và tokenizer
model_name = "CuongPhan23/Chatbot_VietNamese_Law_Qwen"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()

# Đơn giản hóa task manager
active_tasks = {}

class ChatRequest(BaseModel):
    user_message: str
    user_id: str
    bot_id: str

class ChatResponse(BaseModel):
    task_id: str

@app.post("/chat/complete", response_model=ChatResponse)
def generate_task(req: ChatRequest):
    task_id = str(uuid.uuid4())
    prompt = f"<|user|>\n{req.user_message}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    active_tasks[task_id] = inputs
    return {"task_id": task_id}


@app.get("/chat/complete_v2/{task_id}")
def complete_chat(task_id: str):
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    inputs = active_tasks.pop(task_id)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    user_msg = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    reply = generated_text.replace(user_msg, "").strip()

    return {
        "task_result": {
            "content": reply
        }
    }
