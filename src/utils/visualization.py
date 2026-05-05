import matplotlib.pyplot as plt
import os

def plot_loss_curve(train_losses, val_losses, model_name, save_dir="output/figures"):
    # Tự động tạo thư mục nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)
    
    # Thiết lập kích thước khung hình
    plt.figure(figsize=(10, 6))
    
    # Tạo trục X (số lượng Epochs)
    epochs = range(1, len(train_losses) + 1)
    
    # Vẽ đường Train Loss (màu xanh dương, điểm tròn)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    
    # Vẽ đường Validation Loss (màu đỏ, điểm vuông)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
        
    # Làm đẹp biểu đồ
    plt.title(f'Đồ thị hội tụ hàm Loss - Mô hình {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Căn chỉnh và lưu file
    save_path = os.path.join(save_dir, f"{model_name}_loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # dpi=300 để ảnh nét
    plt.close()
    
    print(f"[INFO] Đã lưu biểu đồ thực nghiệm tại: {save_path}")