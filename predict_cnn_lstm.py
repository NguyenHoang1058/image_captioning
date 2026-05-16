import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import warnings

# Ignore các warning
warnings.filterwarnings("ignore")

# Import class Model
from src.models.cnn_lstm_model import EncoderCNN, DecoderRNN

# Import hàm xử lý dữ liệu
from src.data.data_loader import load_data
from src.data.feature_engineering import get_all_captions, build_vocab, build_word2idx, build_idx2word

def predict_image_cnn_lstm(image_path, encoder, decoder, idx2word, transform, device):
    if not os.path.exists(image_path):
        print(f"[LỖI] Không tìm thấy ảnh tại: {image_path}")
        return
    
    try:
        raw_img=Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[LỖI] Không thể đọc ảnh: {e}")
        return
    
    # Tiền xử lý ảnh thành Tensor
    image_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Dự đoán
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        features = encoder(image_tensor)
        # Ép mô hình được phép nói dài 50 từ
        sampled_ids = decoder.generate_caption(features, max_length=50) 

    # Dịch ID thành Text
    pred_words = []
    for word_id in sampled_ids:
        word = idx2word.get(word_id, "<unk>")
        if word == "<end>":
            break
        if word not in ["<start>", "<pad>", "<unk>"]:
            pred_words.append(word)
            
    caption = " ".join(pred_words)

    # Hiển thị
    print("\n" + "="*50)
    print("🎯 KẾT QUẢ DỰ ĐOÁN (CNN-LSTM):")
    print(f"=> {caption.capitalize()}")
    print("="*50 + "\n")

    plt.figure(figsize=(8, 6))
    plt.imshow(raw_img)
    plt.title(f"{caption.capitalize()}", fontsize=14, color='red', fontweight='bold')
    plt.axis('off')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Chạy Inference trên: {device}")

    # --- KHỞI TẠO TỪ ĐIỂN (VOCAB) ---
    print("[INFO] Đang tải từ điển...")
    # Gọi các hàm load data
    train_data, _, _ = load_data()

    # Tái tạo lại từ điển (Vocab)
    captions = get_all_captions(train_data)
    vocab = build_vocab(captions, min_freq=2)
    word2idx = build_word2idx(vocab)
    idx2word = build_idx2word(word2idx)
    vocab_size = len(word2idx)

    embed_size = 256
    hidden_size = 512

    # --- KHỞI TẠO MODEL ---
    print("[INFO] Đang tải trọng số CNN-LSTM...")
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=1).to(device)
    
    # Load trọng số (Sửa lại epoch tốt nhất của bạn)
    encoder.load_state_dict(torch.load("checkpoints/cnn_encoder_epoch_5.pth", weights_only=True))
    decoder.load_state_dict(torch.load("checkpoints/cnn_decoder_epoch_5.pth", weights_only=True))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n--- CÔNG CỤ DỰ ĐOÁN CNN-LSTM ---")
    while True:
        img_path = input("Nhập đường dẫn ảnh (hoặc gõ 'q' để thoát): ")
        if img_path.lower() == 'q':
            break
        img_path = img_path.strip('"').strip("'") 
        predict_image_cnn_lstm(img_path, encoder, decoder, idx2word, transform, device)

if __name__ == "__main__":
    main()