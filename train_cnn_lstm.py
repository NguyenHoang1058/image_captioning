import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import các hàm đã viết ở data_loader và feature_engineering
from src.data.data_loader import load_data
from src.data.preprocessing import prepare_data
from src.data.feature_engineering import get_all_captions, build_vocab, build_word2idx
from src.data.dataset import FlickrDataset

# Import mô hình và vòng lặp huấn luyện
from src.models.cnn_lstm_model import EncoderCNN, DecoderRNN
from src.models.train import train_epoch, validate_epoch

#Import hàm vẽ biểu đồ
from src.utils.visualization import plot_loss_curve

def main():
    # 1. CẤU HÌNH THIẾT BỊ (GPU/CPU)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # 2. CHUẨN BỊ DỮ LIỆU
    print("Đang tải dữ liệu...")
    # Lấy tập train từ thư mục data/processed mà bạn đã lưu
    train_data, val_data, test_data = load_data()

    # Xây dựng từ điển (Vocabulary)
    captions=get_all_captions(train_data)
    vocab=build_vocab(captions, min_freq=2)# Bỏ các từ xuất hiện < 2 lần cho nhẹ model
    word2idx=build_word2idx(vocab)
    vocab_size=len(word2idx)

    # Tiền xử lý tập train (clean, encode)
    processed_train = prepare_data(train_data, word2idx)

    # Khởi tạo data train
    train_dataset = FlickrDataset(processed_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Khởi tạo data để validate
    processed_val = prepare_data(val_data, word2idx)
    val_dataset = FlickrDataset(processed_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 3. KHỞI TẠO MÔ HÌNH (Đây là nơi gọi model)
    embed_size = 256   # Kích thước vector nhúng từ vựng
    hidden_size = 512  # Kích thước bộ nhớ nơ-ron của LSTM
    num_layers = 1     # Số lớp LSTM

    print("Đang khởi tạo mô hình CNN-LSTM...")
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # 4. CẤU HÌNH OPTIMIZER VÀ LOSS FUNCTION
    # Chọn hàm loss, bỏ qua index của token <pad> (thường là 0)
    pad_idx = word2idx["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Khai báo bộ tối ưu hóa Adam
    # Lưu ý: Mạng CNN (ResNet) đã pre-trained nên ta cấp learning rate nhỏ (1e-4) để tránh hỏng tệp trọng số cũ
    # Mạng LSTM train từ đầu nên cấp learning rate lớn hơn (4e-4)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-4)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=4e-4)

    # 5. KÍCH HOẠT VÒNG LẶP HUẤN LUYỆN
    num_epochs = 5 # Chạy thử 5 epochs trước xem kết quả
    print("Bắt đầu huấn luyện...")

    # Khởi tạo 2 mảng rỗng để lưu lịch sử Loss
    train_history = []
    val_history = []

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(
            encoder=encoder, 
            decoder=decoder, 
            dataloader=train_loader, 
            criterion=criterion, 
            encoder_optimizer=encoder_optimizer, 
            decoder_optimizer=decoder_optimizer, 
            vocab_size=vocab_size,
            device=device
        )

        # Validation
        val_loss = validate_epoch(
            encoder=encoder, decoder=decoder, dataloader=val_loader, 
            criterion=criterion, vocab_size=vocab_size, device=device
        )

        # Cập nhật lịch sử sau mỗi epoch
        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"=== Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ===")
        
        # Lưu lại tệp trọng số (weights) sau mỗi epoch để sau này lấy ra Inference (dự đoán)
        torch.save(encoder.state_dict(), f"checkpoints/cnn_encoder_epoch_{epoch}.pth")
        torch.save(decoder.state_dict(), f"checkpoints/cnn_decoder_epoch_{epoch}.pth")

    # Vẽ và lưu biểu đồ khi vòng lặp for kết thúc
    print("\n[INFO] Đang xuất biểu đồ báo cáo...")
    plot_loss_curve(
        train_losses=train_history, 
        val_losses=val_history, 
        model_name="CNN_LSTM"
    )

    print("[INFO] Đã hoàn tất toàn bộ quy trình!")

if __name__ == "__main__":
    main()