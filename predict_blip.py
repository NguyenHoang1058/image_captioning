import torch
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Import model BLIP
from src.models.blip_model import BlipCaptioner

def predict_external_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Kiểm tra và Load ảnh
    if not os.path.exists(image_path):
        print(f"[LỖI] Không tìm thấy ảnh tại đường dẫn: {image_path}")
        return
        
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[LỖI] Không thể đọc file ảnh: {e}")
        return
    
    # 2. Khởi tạo mô hình BLIP + LoRA
    print("[INFO] Đang tải mô hình BLIP + LoRA và trọng số...")
    model = BlipCaptioner(use_lora=True).to(device)

    # Trỏ vào file trọng số
    try:
        model.model.load_adapter("checkpoints/blip_lora/epoch_3", "default")
    except Exception as e:
        print(f"[LỖI] Không tìm thấy file trọng số LoRA: {e}")
        return
        
    model.eval() # Bật chế độ đánh giá

    # 3. Tiến hành sinh câu
    print("[INFO] Đang phân tích bức ảnh...")
    with torch.no_grad():
        caption = model.generate_caption(raw_image)[0]

    # 4. In ra terminal và vẽ lên màn hình
    print("\n" + "="*50)
    print("🎯 KẾT QUẢ DỰ ĐOÁN:")
    print(f"=> {caption.capitalize()}")
    print("="*50 + "\n")

    # Hiển thị ảnh kèm caption bằng Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(raw_image)
    plt.title(caption.capitalize(), fontsize=14, color='blue', fontweight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Cung cấp giao diện nhập đường dẫn ảnh từ Terminal
    print("--- CÔNG CỤ DỰ ĐOÁN IMAGE CAPTIONING (BLIP+LoRA) ---")
    while True:
        img_path = input("Nhập đường dẫn ảnh (hoặc gõ 'q' để thoát): ")
        if img_path.lower() == 'q':
            break
            
        # Xóa dấu ngoặc kép nếu người dùng kéo thả file vào terminal
        img_path = img_path.strip('"').strip("'") 
        predict_external_image(img_path)