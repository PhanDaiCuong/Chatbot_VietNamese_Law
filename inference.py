from unsloth import FastLanguageModel 

from transformers import TextStreamer

model_id = "AIPROENGINEER/Chatbot_VietNamese_Law"  
max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)

alpaca_prompt = """Dưới đây là hướng dẫn mô tả một nhiệm vụ, kết hợp với thông tin đầu vào cung cấp thêm ngữ cảnh. Hãy viết phản hồi hoàn thành yêu cầu một cách phù hợp.
        ### Hướng dẫn:
        Bạn là một trợ lý thông minh, hãy trả lời câu hỏi hiện tại của user dựa trên lịch sử chat và các tài liệu liên quan. Câu trả lời phải ngắn gọn, chính xác nhưng vẫn đảm bảo đầy đủ các ý chính.
        ### Câu hỏi:
        {}

        ### Trả lời:
        {}"""
def generate_text(
    instruction, streaming: bool = True, trim_input_message: bool = False
):
    message = alpaca_prompt.format(
        instruction,
        "",  # output - leave this blank for generation! 
    )
    inputs = tokenizer([message], return_tensors="pt").to("cuda")

    if streaming:
        return model.generate(
            **inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True
        )
    else:
        output_tokens = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        output = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

        if trim_input_message:
            return output[len(message) :]
        else:
            return output
        
print (generate_text("Nhiệm vụ của Hội đồng truyền máu của cơ sở khám bệnh, chữa bệnh là gì?", streaming=True))