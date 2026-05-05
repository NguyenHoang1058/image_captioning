import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import os

# Import các hàm đã viết ở data_loader và feature_engineering
from src.data.data_loader import load_data
from src.data.dataset import BlipDataset, blip_collate_fn

# Import model BLIP đã định nghĩa
from src.models.blip_model import BlipCaptioner

# Import hàm vẽ biểu đồ
from src.utils.visualization import plot_loss_curve

def validate_blip_epoch(model, dataloader):
    model.eval() # Tắt tính năng huấn luyện
    total_loss = 0
    
    with torch.no_grad(): # Tắt lưu trữ đạo hàm để tiết kiệm VRAM
        for images, texts in dataloader:
            # Dùng bfloat16 cho cả lúc validation để tăng tốc
            with autocast(dtype=torch.bfloat16):
                loss = model(images, texts)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    # 1. THIẾT LẬP MÔI TRƯỜNG & THIẾT BỊ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Đang chạy huấn luyện trên thiết bị: {device}")

    # Tạo thư mục lưu model weights nếu chưa có
    os.makedirs("checkpoints/blip", exist_ok=True)

    # 2. CHUẨN BỊ DỮ LIỆU
    print("[INFO] Đang tải dữ liệu Flickr30k...")
    train_data, val_data, test_data = load_data()

    # Bọc data thô vào class BlipDataset
    blip_train_dataset = BlipDataset(train_data)
    val_dataset = BlipDataset(val_data)

    # Khởi tạo DataLoader. Bắt buộc dùng collate_fn custom để batching ảnh PIL và String
    # Batch size để 8 hoặc 16 tùy thuộc vào VRAM (BLIP khá nặng)
    train_loader = DataLoader(
        blip_train_dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=blip_collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=blip_collate_fn,
        num_workers=0,
        pin_memory=True
    )

    print(f"[INFO] Tổng số batch trong 1 epoch: {len(train_loader)}")

    # 3. KHỞI TẠO MÔ HÌNH
    print("[INFO] Khởi tạo BLIP Model (Tải pre-trained weights từ HuggingFace)...")
    model = BlipCaptioner(use_lora=True).to(device)

    # Optimizer dùng AdamW, lr set 1e-4 phù hợp với LoRA
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 5. VÒNG LẶP HUẤN LUYỆN (FINE-TUNING)
    num_epochs = 3 # Transformer hội tụ rất nhanh, 3 epochs thường là đủ cho đồ án
    accumulation_steps = 2
    train_history = []
    val_history = []
    print("[INFO] Bắt đầu quá trình Fine-tuning...")

    for epoch in range(1, num_epochs + 1):
        model.train() # Set mode train
        total_train_loss = 0
        optimizer.zero_grad()
        
        for step, (images, texts) in enumerate(train_loader):
            with autocast(dtype=torch.bfloat16):
                loss = model(images, texts)
                loss = loss/accumulation_steps
                       
            # Backpropagation
            loss.backward()
            
            # Cập nhật trọng số
            if (step+1)%accumulation_steps==0 or (step+1)==len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps # Trả lại scale gốc để hiển thị
            
            # In log mỗi 50 steps để theo dõi tiến độ
            if step % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] | Step [{step}/{len(train_loader)}] | Loss: {loss.item() * accumulation_steps:.4f}")

        # Tính toán loss trung bình của Epoch
        avg_train_loss=total_train_loss/len(train_loader)
        # Gọi hàm validation
        avg_val_loss = validate_blip_epoch(model, val_loader)

        # Lưu lịch sử để vẽ biểu đồ
        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)

        print(f"=== KẾT THÚC EPOCH {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ===")
        
        # 6. LƯU CHECKPOINT
        # Lưu lại state_dict để sau này có thể load lên Inference hoặc đánh giá (BLEU/METEOR)
        adapter_save_path = f"checkpoints/blip_lora/epoch_{epoch}"
        model.model.save_pretrained(adapter_save_path)
        print(f"[INFO] Đã lưu Adapter tại {adapter_save_path}\n")

    
    print("\n[INFO] Đang xuất biểu đồ báo cáo...")
    plot_loss_curve(
        train_losses=train_history, 
        val_losses=val_history, 
        model_name="BLIP"
    )

    print("[INFO] Hoàn thành huấn luyện BLIP!")

if __name__ == "__main__":
    main()