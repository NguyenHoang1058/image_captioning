import torch
import json
import os
from tqdm import tqdm

# Import module load data
from src.data.data_loader import load_data

# Import model BLIP
from src.models.blip_model import BlipCaptioner

# Import mode CNN_LSTM và ATTENTION
from src.models.cnn_lstm_model import EncoderCNN, DecoderRNN
from src.models.attention_model import EncoderAttention, DecoderAttention

def run_inference(model_name, encoder, decoder, test_data, device, vocab=None):
    print(f"\n[INFO] Đang sinh câu dự đoán cho mô hình: {model_name}...")

    # Set mode eval cho tất cả
    if encoder: encoder.eval()
    if decoder: decoder.eval()
    results = []

    with torch.no_grad():
        for item in tqdm(test_data):
            img_path = item['image_path']
            raw_img = item['image']
            true_caption = item['caption']

            if "BLIP" in model_name:
                pred_caption = encoder.generate_caption(raw_img)[0]
            else:
                image_tensor = item['tensor_image'].to(device)

                # Đưa qua Encoder để lấy đặc trưng
                features = encoder(image_tensor)

                # Đưa qua Decoder để sinh câu
                sampled_ids = decoder.sample(features)

                # Dịch ID số thành chữ
                pred_caption = "câu dự đoán của bạn"

            results.append({
                "image": img_path,
                "reference": true_caption,
                "prediction": pred_caption
            })

    # Lưu ra file JSON
    os.makedirs("results", exist_ok=True)
    save_path = f"results/{model_name}_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Đã lưu kết quả tại: {save_path}")