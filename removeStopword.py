import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

df = pd.read_csv('data/full_train_1.csv')

# Đọc file stopwords
with open('data/vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
    vietnamese_stopwords = f.read().splitlines()

# Tạo danh sách các stopwords
stop_words = set(vietnamese_stopwords)

# Hàm để loại bỏ stopwords
def remove_stopwords(text):
    if not isinstance(text, str):
        return text  # Trả lại giá trị ban đầu nếu không phải là chuỗi
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tách từ
    words = text.split()
    # Loại bỏ stopwords
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


# Áp dụng hàm loại bỏ stopwords vào cột Comment
df['Comment'] = df['Comment'].apply(remove_stopwords)

# Lưu lại file CSV đã xử lý
df.to_csv('data/full_train_1_cleaned.csv', index=False)
