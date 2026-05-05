import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        # Gọi constructor của class cha do đang kế thừa: nn.modules
        super(EncoderCNN, self).__init__()

        # Load mô hình ResNet50 đã được train sẵn trên tập ImageNet
        resnet=models.resnet50(pretrained=True)

        # Cắt bỏ lớp FullyConnected (fc) cuối cùng vì không cần phân loại ảnh thành 1000 lớp
        # Chỉ cần lấy vector đặc trưng (feature vector) của bức ảnh
        modules=list(resnet.children())[:-1]
        self.resnet=nn.Sequential(*modules)

        # Ánh xạ từ 2048 chiều (chuẩn ResNet) xuống embed_size
        self.linear=nn.Linear(resnet.fc.in_features, embed_size)
        self.bn=nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # Input: (batch_size, 3, 224, 224)
        with torch.no_grad():
            features=self.resnet(images) # Output: (batch_size, 2048, 1, 1)

        # Duỗi ma trận 4D thành ma trận 2D
        features=features.view(features.size(0), -1) # Output: (batch_size, 2048)

        # Đưa qua Linear layer và Batch Normalization để chuẩn hóa
        features=self.bn(self.linear(features)) # Output cuối: (batch_size, embed_size)

        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # Lớp Embedding chuyển đổi số index của từ thành một vector số thực có ý nghĩa
        self.embed=nn.Embedding(vocab_size, embed_size)

        # batch_first=True giúp tensor input có dạng (batch, seq_len, features)
        self.lstm=nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # Lớp ánh xạ từ state của LSTM ra lại bộ từ vựng để dự đoán từ tiếp theo
        self.linear=nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Captions truyền vào đã loại bỏ token <end>
        # Input captions shape: (batch_size, seq_length - 1)
        embeddings=self.embed(captions)

        # Nối vector ảnh vào vị trí ĐẦU TIÊN của chuỗi để "nhắc" LSTM bức ảnh này nói về gì
        # features.unsqueeze(1) biến (batch_size, embed_size) thành (batch_size, 1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        # Shape sau khi nối: (batch_size, seq_length, embed_size)

        # Đẩy toàn bộ chuỗi qua LSTM trong 1 lần (cơ chế teacher forcing)
        hiddens, _ = self.lstm(embeddings) # hiddens shape: (batch_size, seq_length, hidden_size)
        # Đưa ra dự đoán xác suất cho mỗi từ trong vocab tại mỗi bước thời gian
        outputs = self.linear(hiddens) # Shape: (batch_size, seq_length, vocab_size)
        return outputs
    
    def generate_caption(self, features, max_length=20, start_idx=1, end_idx=2):
        # --- DÙNG KHI ĐÁNH GIÁ (INFERENCE) ---
        captions = []
        inputs = features.unsqueeze(1) # (batch_size=1, 1, embed_size)
        states = None 
        
        for _ in range(max_length):
            # LSTM xử lý từng bước
            hiddens, states = self.lstm(inputs, states) 
            outputs = self.linear(hiddens.squeeze(1)) # (1, vocab_size)
            
            # Lấy từ có xác suất cao nhất
            predicted = outputs.argmax(1) 
            captions.append(predicted.item())
            
            # Dừng nếu gặp token <end>
            if predicted.item() == end_idx:
                break
                
            # Cập nhật đầu vào cho bước tiếp theo bằng chính từ vừa sinh ra
            inputs = self.embed(predicted).unsqueeze(1) 
            
        return captions