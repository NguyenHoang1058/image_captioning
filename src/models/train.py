import torch
import torch.nn as nn

def train_epoch(encoder, decoder, dataloader, criterion, encoder_optimizer, decoder_optimizer, vocab_size, device):
    # Đặt mode sang train để kích hoạt Dropout và BatchNorm (rất quan trọng)
    encoder.train()
    decoder.train()
    total_loss=0

    for idx,(images, captions) in enumerate(dataloader):
        # Đưa dữ liệu lên VRAM của GPU (nếu có) để tăng tốc ma trận
        images=images.to(device)
        captions=captions.to(device)

        # Xóa các giá trị đạo hàm (gradient) từ batch trước đó
        # Nếu không xóa, PyTorch sẽ cộng dồn đạo hàm làm hỏng trọng số
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # --- FORWARD PASS ---
        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])

        if outputs.size(1) == captions.size(1):
            targets=captions # Dùng cho mô hình cnn_lstm
        else:
            targets=captions[:, 1:] # Dùng cho mô hình Attention

        # --- CALCULATE LOSS ---
        # Hàm loss của PyTorch nhận đầu vào 2D, nên ta phải nén batch và seq_length lại
        # outputs shape: (batch_size * seq_length, vocab_size)
        # targets shape: (batch_size * seq_length)
        loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.contiguous().view(-1))

        # --- BACKWARD PASS ---
        # Tính toán đạo hàm ngược qua toàn bộ computational graph
        loss.backward()

        # Cập nhật trọng số của mạng (Weights & Biases) dựa trên learning rate
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

        # In log ra terminal để theo dõi quá trình chạy
        if idx % 100 == 0:
            print(f"Step [{idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def validate_epoch(encoder, decoder, dataloader, criterion, vocab_size, device):
    # Đưa mô hình về chế độ đánh giá (tắt Dropout và BatchNorm)
    encoder.eval()
    decoder.eval()
    total_loss = 0

    # Tắt tính toán đạo hàm (Gradient) để tiết kiệm VRAM và chạy nhanh hơn
    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            
            if outputs.size(1) == captions.size(1):
                targets=captions # Dùng cho mô hình cnn_lstm
            else:
                targets=captions[:, 1:] # Dùng cho mô hình Attention

            # Tính loss
            loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.contiguous().view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)