import matplotlib.pyplot as plt
from collections import Counter
from src.data.preprocessing import clean_text
from PIL import Image
import random
import os
from IPython.display import display

# Trích xuất toàn bộ caption từ dataset thành danh sách để phục vụ phân tích
def get_all_caption(data):
    captions=[]
    for item in data:
        captions.extend(item["caption"])
    return captions

# Vẽ biểu đồ histogram độ dài caption để phân tích phân phối số lượng từ
def plot_caption_length(captions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lengths=[len(cap.split()) for cap in captions]

    plt.figure()
    plt.hist(lengths, bins=20)
    plt.title("Caption Length Distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")

    plt.savefig(save_path)
    plt.close()

# Vẽ biểu đồ các từ xuất hiện nhiều nhất nhằm xác định nội dung phổ biến trong dataset
def plot_top_words(captions, save_path, top_n=10):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    words=[]

    for cap in captions:
        words.extend(clean_text(cap).split())

    counter=Counter(words)
    common=counter.most_common(top_n)

    labels=[x[0] for x in common]
    values=[x[1] for x in common]

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.title("Top Words")

    plt.savefig(save_path)
    plt.close()

# Hiển thị một số ảnh và caption tương ứng để quan sát trực quan dữ liệu
def show_sample(data):
    sample=random.choice(data)

    idx=random.randint(0, len(data)-1)
    sample = data[idx]

    print(sample["caption"])
    display(sample["image"])

# Vẽ histogram số lượng từ không trùng lặp trong mỗi caption để đánh giá độ phong phú từ vựng
def plot_unique_words_per_caption(captions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    unique_counts=[len(set(c.split())) for c in captions]

    plt.hist(unique_counts, bins=30)
    plt.title("Unique Words per Caption")
    plt.xlabel("Number of Unique Words")
    plt.ylabel("Frequency")

    plt.savefig(save_path)
    plt.close()

# Tạo tập hợp các từ vựng duy nhất (vocabulary) từ toàn bộ caption
def get_vocab(captions):
    words=[]
    for c in captions:
        words.extend(c.lower().split())
    return set(words)

# Vẽ biểu đồ tần suất xuất hiện của các từ phổ biến trong dataset
def plot_word_frequency(captions, save_path):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    words = []
    for c in captions:
        words.extend(c.lower().split())

    counter = Counter(words)
    most_common = counter.most_common(20)

    words, freqs = zip(*most_common)

    import matplotlib.pyplot as plt
    plt.bar(words, freqs)
    plt.xticks(rotation=45)
    plt.title("Top 20 Most Frequent Words")

    plt.savefig(save_path)
    plt.close()

# Hàm tính TTR cho từng caption (tỷ lệ số từ khác nhau trên tổng số từ)
# và vẽ histogram để thể hiện phân phối độ đa dạng từ vựng.
def plot_ttr_distribution(captions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ttr_values = []
    for c in captions:
        words = c.split()
        if len(words) > 0:
            ttr = len(set(words)) / len(words)
            ttr_values.append(ttr)

    plt.hist(ttr_values, bins=30)
    plt.title("TTR Distribution")
    plt.xlabel("TTR")
    plt.ylabel("Frequency")

    plt.savefig(save_path)
    plt.close()

# Tìm ảnh theo từ khóa
def find_samples_by_keyword(data, keyword, n=1):
    results = []
    for item in data:
        if keyword in " ".join(item["caption"]).lower():
            results.append(item)
        if len(results) == n:
            break
    return results