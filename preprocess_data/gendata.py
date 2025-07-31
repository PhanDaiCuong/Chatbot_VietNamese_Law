
import os
import pandas as pd
import ast
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import tqdm
import argparse # Thư viện để đọc tham số dòng lệnh
import time 
from google.api_core.exceptions import ResourceExhausted

# --- PHẦN 1: CẤU HÌNH VÀ HÀM API (Đã tối ưu) ---

def setup_api():
    """Nạp API key từ .env và cấu hình an toàn."""
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Không tìm thấy GOOGLE_API_KEY. Vui lòng tạo file .env.")
    genai.configure(api_key=api_key)

def create_gemini_model():
    """Tạo và trả về một instance của model Gemini."""
    generation_config = {
      "temperature": 0.2,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 4096,
    }
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        generation_config=generation_config
    )
    return model

def gen_answer(model, query, document):
    """
    Hàm gọi API để sinh câu trả lời, có tích hợp tự động thử lại khi gặp lỗi 429.
    """
    prompt = f"""
    Là một trợ lý ảo thông minh, hãy trả lời câu hỏi sau đây chỉ dựa trên nội dung trong phần "Tài liệu".
    Câu trả lời phải ngắn gọn, chính xác, và đi thẳng vào vấn đề. Nếu thông tin không có trong tài liệu, hãy nói "Thông tin không có trong tài liệu được cung cấp".

    # Question:
    {query}

    # Tài liệu:
    {document}
    """
    
    retry_count = 0
    max_retries = 5  # Sẽ thử lại tối đa 5 lần
    wait_time = 30 # Thời gian chờ ban đầu là 45 giây (an toàn hơn 30)

    while retry_count < max_retries:
        try:
            response = model.generate_content(prompt)
            return response.text.strip() # Nếu thành công, trả về kết quả và thoát khỏi vòng lặp
        
        # Bắt đúng lỗi 429 (ResourceExhausted) của Google
        except ResourceExhausted as e:
            retry_count += 1
            print(f"\nLỗi Rate Limit (429). Nguyên nhân có thể do RPM hoặc TPM.")
            print(f"Đang là lần thử lại thứ {retry_count}/{max_retries}. Chờ {wait_time} giây...")
            time.sleep(wait_time)
            wait_time *= 1.5 

        # Bắt các lỗi khác và trả về ngay lập tức
        except Exception as e:
            print(f"\nĐã xảy ra một lỗi API không xác định: {e}")
            return "Lỗi khi gọi API"
            
    print(f"\nVượt quá số lần thử lại tối đa cho câu hỏi: '{query[:50]}...'")
    return "Lỗi: Vượt quá giới hạn API sau nhiều lần thử"

# --- PHẦN 2: LOGIC XỬ LÝ DỮ LIỆU (Đã tối ưu) ---

def main(args):
    """Hàm chính để chạy toàn bộ tiến trình."""
    # Thiết lập API và Model
    setup_api()
    model = create_gemini_model()

    # Đọc dữ liệu đầu vào
    try:
        df_input = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file đầu vào '{args.input_file}'")
        return

    # Đọc file output nếu có để nối lại công việc
    if os.path.exists(args.output_file):
        df_output = pd.read_csv(args.output_file)
        # Tạo một set các câu hỏi đã xử lý để kiểm tra nhanh hơn
        processed_questions = set(df_output['question'])
        print(f"Đã tìm thấy file output. Đã xử lý {len(processed_questions)} câu hỏi. Sẽ tiếp tục.")
    else:
        df_output = pd.DataFrame(columns=["question", "context", "answer"])
        processed_questions = set()
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)


    # **CẢI TIẾN HIỆU NĂNG**: Dùng list để lưu kết quả tạm thời
    results_list = []

    # Giới hạn vòng lặp theo tham số đầu vào
    start_index = args.start
    end_index = args.end if args.end else len(df_input)

    print(f"Bắt đầu xử lý từ dòng {start_index} đến {end_index}...")

    # Sử dụng tqdm để theo dõi tiến trình
    for index, row in tqdm.tqdm(df_input.iloc[start_index:end_index].iterrows(), total=end_index-start_index):
        question = row['question']

        # Bỏ qua nếu câu hỏi đã được xử lý (Logic này vẫn đúng)
        if question in processed_questions:
            continue

        try:
            # 1. XỬ LÝ DỮ LIỆU
            contexts = ast.literal_eval(row['context'])
            full_context = "Dưới đây là toàn bộ thông tin về tài liệu : \n" + "\n".join(contexts) + "\nKết thúc phần thông tin tài liệu."
            
            # 2. GỌI API ĐỂ SINH CÂU TRẢ LỜI
            answer = gen_answer(model, question, full_context)
            
            # Nếu API trả về lỗi, câu trả lời sẽ là chuỗi lỗi, ta không lưu nó
            if "Lỗi khi gọi API" in answer or "Vượt quá giới hạn" in answer:
                print(f"Bỏ qua dòng {index} do lỗi API không thể khắc phục.")
                continue # Bỏ qua và sang dòng tiếp theo

            # 3. THÊM KẾT QUẢ THÀNH CÔNG VÀO LIST
            results_list.append({"question": question, "context": full_context, "answer": answer})
            time.sleep(10)
            # 4. LƯU FILE ĐỊNH KỲ (ĐẶT Ở ĐÂY LÀ ĐÚNG)
            # Chỉ lưu khi có kết quả mới và đã đến lúc
            if results_list and len(results_list) % args.save_every == 0:
                # Tạo DataFrame từ các kết quả mới tích lũy được
                temp_df = pd.DataFrame(results_list)
                # Nối với DataFrame output cũ
                df_to_save = pd.concat([df_output, temp_df], ignore_index=True)
                # Lưu xuống file
                df_to_save.to_csv(args.output_file, index=False, encoding="utf-8-sig")
                print(f"\nCheckpoint: Đã lưu {len(df_to_save)} dòng vào {args.output_file}")
                
                # Sau khi lưu, xóa list tạm để không bị lưu trùng lặp
                # và cập nhật df_output
                df_output = df_to_save
                results_list = []



        # GỘP LẠI THÀNH MỘT KHỐI EXCEPT DUY NHẤT
        except Exception as e:
            print(f"\nLỗi nghiêm trọng khi xử lý dòng {index}: {e}")
            print("Bỏ qua dòng này và tiếp tục...")
            continue

    # Lưu nốt phần còn lại sau khi vòng lặp kết thúc
    if results_list:
        final_temp_df = pd.DataFrame(results_list)
        df_final = pd.concat([df_output, final_temp_df], ignore_index=True)
        df_final.to_csv(args.output_file, index=False, encoding="utf-8-sig")
        print(f"\nHoàn tất! Đã lưu tổng cộng {len(df_final)} dòng vào file {args.output_file}")
if __name__ == "__main__":
    # --- PHẦN 3: THAM SỐ DÒNG LỆNH (Linh hoạt hơn) ---
    parser = argparse.ArgumentParser(description="Sinh câu trả lời từ file CSV bằng Gemini API.")
    parser.add_argument('--input_file', type=str, default='/home/cuong/PycharmProjects/Transformer/Vietnamese-Law-Question-Answering-system/train.csv', help='Đường dẫn đến file CSV đầu vào.')
    parser.add_argument('--output_file', type=str, default='gen_data/train_generated.csv', help='Đường dẫn đến file CSV đầu ra.')
    parser.add_argument('--start', type=int, default=0, help='Index dòng bắt đầu xử lý.')
    parser.add_argument('--end', type=int, default=None, help='Index dòng kết thúc xử lý (không bao gồm). Bỏ trống để chạy đến hết file.')
    parser.add_argument('--save-every', type=int, default=50, help='Lưu file sau mỗi N dòng xử lý.')
    
    args = parser.parse_args()
    main(args)


