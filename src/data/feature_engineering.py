from collections import Counter

# Lấy toàn bộ caption từ dataset (gộp thành 1 list)
def get_all_captions(data):
    captions = []

    for item in data:
        if isinstance(item["caption"], list):
            captions.extend(item["caption"])
        else:
            captions.append(item["caption"])

    return captions

# Tách câu thành các từ (tokenization đơn giản)
def tokenize(caption):
    return caption.split()

# Xây dựng vocabulary từ tất cả caption
# min_freq: chỉ giữ các từ xuất hiện >= min_freq
def build_vocab(captions, min_freq=1):
    counter = Counter()

    for cap in captions:
        tokens = tokenize(cap)
        counter.update(tokens)

    vocab = [word for word, freq in counter.items() if freq >= min_freq]

    return vocab

# Tạo mapping từ → index
# Bao gồm các token đặc biệt dùng trong mô hình
def build_word2idx(vocab):
    word2idx = {
        "<pad>": 0,   # padding
        "<start>": 1, # bắt đầu câu
        "<end>": 2,   # kết thúc câu
        "<unk>": 3    # từ không có trong vocab
    }

    for i, word in enumerate(vocab, start=4):
        word2idx[word] = i

    return word2idx

# Tạo mapping ngược từ index → word
# Dùng để decode output của model
def build_idx2word(word2idx):
    return {idx: word for word, idx in word2idx.items()}

# Chuyển 1 caption → danh sách số (index)
# Thêm <start> và <end> để mô hình học sequence
def encode_caption(caption, word2idx):
    tokens = tokenize(caption)

    encoded = [word2idx["<start>"]]

    for token in tokens:
        encoded.append(word2idx.get(token, word2idx["<unk>"]))

    encoded.append(word2idx["<end>"])

    return encoded

# Chuẩn hóa độ dài caption
# Nếu ngắn → thêm <pad>, nếu dài → cắt bớt
def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# Encode toàn bộ dataset
# Kết quả: thêm trường "encoded_caption" vào mỗi sample
def encode_dataset(data, word2idx, max_len=20):
    new_data = []

    for item in data:
        new_item = item.copy()

        captions = new_item["caption"]

        if isinstance(captions, list):
            encoded_caps = [
                pad_sequence(encode_caption(c, word2idx), max_len)
                for c in captions
            ]
        else:
            encoded_caps = pad_sequence(
                encode_caption(captions, word2idx), max_len
            )

        new_item["encoded_caption"] = encoded_caps
        new_data.append(new_item)

    return new_data