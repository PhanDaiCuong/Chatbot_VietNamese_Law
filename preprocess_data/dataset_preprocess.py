from datasets import load_dataset, Dataset, concatenate_datasets
import re
import pandas as pd
from huggingface_hub import whoami
from datasets import Features, Value, Sequence
import ast
from datasets import DatasetDict


# Hàm làm sạch context
def clean_and_normalize_legal_text(text: str) -> str:
    # Xoá thẻ <jsontable ...> và </jsontable> nhưng giữ nội dung bên trong
    text = re.sub(r'<jsontable[^>]*>', '', text)
    text = re.sub(r'</jsontable>', '', text)

    # Thay thế các ký tự đặc biệt
    text = text.replace('\xa0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Ghép các dòng không kết thúc bằng dấu câu và không bắt đầu bằng chỉ mục mới
    lines = text.split('\n')
    merged_lines = []
    buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            merged_lines.append("")  # Giữ dòng trống
            continue

        is_new_block = re.match(r'^(Điều\s+\d+|[-•\d]+\)|\d+\.)', stripped)

        if buffer:
            if not re.search(r'[.;:!?…”"]$', buffer) and not is_new_block:
                buffer += ' ' + stripped
            else:
                merged_lines.append(buffer)
                buffer = stripped
        else:
            buffer = stripped

    if buffer:
        merged_lines.append(buffer)

    return '\n'.join(merged_lines)

# Hàm map tổng hợp
def preprocess2(example):
    title = example["question"]

    # Rewrite question từ tiêu đề
    if title.lower().startswith("nghị quyết"):
        new_q = f"Nội dung của {title} là gì?"
    elif title.lower().startswith("luật"):
        new_q = f"{title} quy định những gì?"
    elif title.lower().startswith("báo cáo"):
        new_q = f"{title} phản ánh điều gì?"
    else:
        new_q = f"Nội dung của văn bản '{title}' là gì?"

    # Clean context nếu là string
    cleaned_context = clean_and_normalize_legal_text(example["context"])
    example["question"] = new_q
    example["context"] = [cleaned_context]
    return example


def fix_context(example):
    if isinstance(example['context'], str):
        try:
            example['context'] = ast.literal_eval(example['context'])
        except Exception:
            example['context'] = [example['context']]
    return example



if __name__ == "__main__":
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds1_train= load_dataset("YuITC/Vietnamese-Legal-Doc-Retrieval-Data", split = "train")
    ds1_test= load_dataset("YuITC/Vietnamese-Legal-Doc-Retrieval-Data", split = "test")


    ds1_train = ds1_train.rename_columns({
        "context_list":"context"
    })
    ds1_test = ds1_test.rename_columns({
        "context_list":"context"
    })


    ds1_train = ds1_train.remove_columns([col for col in ds1_train.column_names if col not in ["question", "context"]])

    ds1_test = ds1_test.remove_columns([col for col in ds1_test.column_names if col not in ["question", "context"]])


    ds2_train = load_dataset("sontungkieu/ThuVienPhapLuat", split = "train")



    ds2_train= ds2_train.remove_columns([col for col in ds2_train.column_names if col not in ["title", "noi_dung"]])


    ds2_train = ds2_train.rename_columns({
        "title":"question",
        "noi_dung":"context"
    })

    ds2_train_clean = ds2_train.map(preprocess2)

    ds3_train = pd.read_csv("/home/cuong/AI_Projects/Chatbot_VietNamese_Law/Law_data/train_split.csv")
    ds3_val = pd.read_csv("/home/cuong/AI_Projects/Chatbot_VietNamese_Law/Law_data/val_split.csv")
    ds3_test = pd.read_csv("/home/cuong/AI_Projects/Chatbot_VietNamese_Law/Law_data/public_test.csv")



    ds3_train = ds3_train[["question", "context"]]
    ds3_val = ds3_val[["question", "context"]]
    ds3_test = ds3_test[["question"]]



    ds3_train = Dataset.from_pandas(ds3_train)
    ds3_val = Dataset.from_pandas(ds3_val)
    ds3_test = Dataset.from_pandas(ds3_test)


    features = Features({
        "question": Value("string"),
        "context": Sequence(Value("string"))
    })
    ds3_test = ds3_val.cast(features)


    ds3_train = ds3_train.map(fix_context)
    ds3_val = ds3_val.map(fix_context)
    ds3_test = ds3_test.map(fix_context)



    features = Features({
        "question": Value("string"),
        "context": Sequence(Value("string"))
    })

    ds1_train = ds1_train.cast(features)
    # ds2_train_clean = ds2_train_clean.cast(features)
    ds3_train= ds3_train.cast(features)

    ds1_test = ds1_test.cast(features)
    ds3_test = ds3_val.cast(features)

    merged_dataset_train = concatenate_datasets([ds1_train , ds3_train])

    merged_dataset_test = concatenate_datasets([ds1_test, ds3_test])


    dataset_dict = DatasetDict({
        "train": merged_dataset_train,
        "validation": merged_dataset_test,
    })


    dataset_dict.push_to_hub("PhanDai/luat-viet-nam-qa_small")

