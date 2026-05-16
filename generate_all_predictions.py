import torch
import json
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Import module load data
from src.data.data_loader import load_data
from src.data.feature_engineering import get_all_captions, build_vocab, build_word2idx, build_idx2word

# Import model BLIP
from src.models.blip_model import BlipCaptioner

# Import mode CNN_LSTM và ATTENTION
from src.models.cnn_lstm_model import EncoderCNN, DecoderRNN
from src.models.attention_model import EncoderAttention, DecoderAttention

def run_inference(model_name, encoder, decoder, test_data, device, idx2word=None):
    print(f"\n[INFO] Đang sinh câu dự đoán cho mô hình: {model_name}...")

    # Set mode eval cho tất cả
    if encoder: encoder.eval()
    if decoder: decoder.eval()
    results = []

    # Hàm biến đổi ảnh dành riêng cho CNN-LSTM và Attention
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for idx, item in enumerate(tqdm(test_data)):
            raw_img=item['image']

            # Đảm bảo ảnh ở hệ màu RGB (tránh lỗi với ảnh đen trắng)
            if raw_img.mode != 'RGB':
                raw_img = raw_img.convert('RGB')

            true_caption = item['caption']

            if "BLIP" in model_name:
                pred_caption = encoder.generate_caption(raw_img)[0]
            else:
                image_tensor = transform(raw_img).unsqueeze(0).to(device)

                # Đưa qua Encoder để lấy đặc trưng
                features = encoder(image_tensor)

                # Đưa qua Decoder để sinh câu
                if "CNN" in model_name:
                    sampled_ids = decoder.generate_caption(features)
                else:
                    sampled_ids = decoder.generate_caption(features)

                # Dịch ID số thành chữ
                pred_words = []
                for word_id in sampled_ids:
                    word = idx2word.get(word_id, "<unk>")
                    if word == "<end>":
                        break
                    if word not in ["<start>", "<pad>", "<unk>"]:
                        pred_words.append(word)
                pred_caption = " ".join(pred_words)

            results.append({
                "image_id": f"test_img_{idx}",
                "reference": true_caption,
                "prediction": pred_caption
            })

    # Lưu ra file JSON
    os.makedirs("results", exist_ok=True)
    save_path = f"results/{model_name}_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Đã lưu kết quả tại: {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu Inference trên: {device}")

    train_data, val_data, test_data = load_data()

    # Tái tạo lại từ điển (Vocab)
    captions = get_all_captions(train_data)
    vocab = build_vocab(captions, min_freq=2)
    word2idx = build_word2idx(vocab)
    idx2word = build_idx2word(word2idx)
    vocab_size = len(word2idx)

    embed_size = 256
    hidden_size = 512

    # CHẠY CNN-LSTM
    print("\n[INFO] Tải CNN-LSTM...")
    cnn_enc = EncoderCNN(embed_size).to(device)
    cnn_dec = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=1).to(device)
    cnn_enc.load_state_dict(torch.load("checkpoints/cnn_encoder_epoch_5.pth", weights_only=True))
    cnn_dec.load_state_dict(torch.load("checkpoints/cnn_decoder_epoch_5.pth", weights_only=True))

    run_inference("CNN_LSTM", cnn_enc, cnn_dec, test_data, device, idx2word)

    # Xóa mô hình khỏi RAM và dọn dẹp VRAM của GPU sau khi hoàn tất thực thi
    del cnn_enc
    del cnn_dec
    torch.cuda.empty_cache() 
    print("[INFO] Đã giải phóng VRAM của CNN-LSTM.")

    # CHẠY ATTENTION
    print("\n[INFO] Tải ATTENTION...")
    att_enc = EncoderAttention().to(device)
    att_dec = DecoderAttention(embed_size, vocab_size, 256, 2048, hidden_size).to(device)
    att_enc.load_state_dict(torch.load("checkpoints/attn_encoder_epoch_5.pth", weights_only=True)) 
    att_dec.load_state_dict(torch.load("checkpoints/attn_decoder_epoch_5.pth", weights_only=True))

    run_inference("ATTENTION", att_enc, att_dec, test_data, device, idx2word)

    # Xóa mô hình khỏi RAM và dọn dẹp VRAM của GPU sau khi hoàn tất thực thi
    del att_enc
    del att_dec
    torch.cuda.empty_cache()
    print("[INFO] Đã giải phóng VRAM của ATTENTION.")

    # CHẠY BLIP + LoRA
    print("[INFO] Tải BLIP + LoRA...")
    blip_lora = BlipCaptioner(use_lora=True).to(device)
    blip_lora.model.load_adapter("checkpoints/blip_lora/epoch_3", "default")

    run_inference("BLIP_LoRA", blip_lora, None, test_data, device)

    # Xóa mô hình khỏi RAM và dọn dẹp VRAM của GPU sau khi hoàn tất thực thi
    del blip_lora
    torch.cuda.empty_cache()
    print("[INFO] Hoàn thành toàn bộ tiến trình sinh dự đoán!")

if __name__ == "__main__":
    main()