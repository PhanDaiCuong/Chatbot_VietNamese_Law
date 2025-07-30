import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import tqdm
import google.generativeai as genai

# Load API key từ .env
_ = load_dotenv(find_dotenv())
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Không tìm thấy GOOGLE_API_KEY. Vui lòng tạo file .env.")
genai.configure(api_key=api_key)

# Load mô hình Gemini Pro
model = genai.GenerativeModel("gemini-pro")

# Load dữ liệu kết quả
df = pd.read_csv("/home/ivirse/namnt/final_project/gen_data/results.csv")

list_score = []

for index, row in tqdm.tqdm(df.iterrows()):
    if index > 1000:
        break

    question = row['question']
    bot_answer = row['bot_answer']
    ground_truth = row['grouth_truth']

    # Tạo prompt đánh giá
    prompt = f"""
    Câu hỏi: {question}
    
    Câu trả lời của mô hình: {bot_answer}
    
    Câu trả lời đúng: {ground_truth}
    
    Hãy đánh giá độ đúng của câu trả lời của mô hình so với câu trả lời đúng trên thang điểm 0 đến 1.
    - Trả về duy nhất một số trong khoảng từ 0 đến 1.
    - Không cần giải thích, chỉ in ra số.
    """

    try:
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        score = float(score_text)
        if 0 <= score <= 1:
            list_score.append(score)
        else:
            print(f"⚠️ Điểm không hợp lệ tại dòng {index}: {score_text}")
    except Exception as e:
        print(f"❌ Lỗi tại dòng {index}: {e}")
        continue

# Tính điểm trung bình
if list_score:
    final_score = sum(list_score) / len(list_score)
    print(f"\n✅ Final Score: {final_score:.4f}")
else:
    print("⚠️ Không có điểm nào được tính.")
