import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

class EncoderAttention(nn.Module):
    def __init__(self):
        super(EncoderAttention, self).__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Ở đây ta cắt bỏ 2 lớp cuối (Global Average Pooling và Fully Connected)
        # Mục đích là giữ lại cấu trúc không gian (spatial) của bức ảnh, không gộp lại thành 1 cục
        modules=list(resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)

    def forward(self, images):
        # Input shape: (batch_size, 3, 224, 224)
        features=self.resnet(images) # Output: (batch_size, 2048, 7, 7)

        # Hoán đổi trục để đưa channel về cuối: (batch_size, 7, 7, 2048)
        features=features.permute(0, 2, 3, 1)

        # Trải phẳng 7x7 thành 49 khu vực (tương đương 49 pixel đại diện cho ảnh)
        # Điều này giúp Attention tính điểm cho 49 khu vực này độc lập
        features=features.view(features.size(0), -1, features.size(-1))
        
        # Output cuối: (batch_size, 49, 2048)
        return features
    
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        # Khởi tạo các mạng Neural nhỏ để tính điểm số Attention
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # Tính điểm từ phần hình ảnh
        att1 = self.encoder_att(encoder_out) # (batch, 49, att_dim)
        # Tính điểm từ state hiện tại của LSTM (từ đang sinh ra)
        att2 = self.decoder_att(decoder_hidden) # (batch, att_dim)
        
        # Cộng gộp và cho qua activation function
        att = self.full_att(torch.relu(att1 + att2.unsqueeze(1))).squeeze(2) # (batch, 49)
        
        # Softmax để chuẩn hóa tổng các điểm bằng 1
        alpha = self.softmax(att) # Trọng số attention cho 49 pixel
        
        # Nhân ma trận trọng số với vector hình ảnh gốc để tạo Context Vector
        # Nghĩa là model quyết định đang "nhìn" tập trung vào đâu trên ảnh
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (batch, 2048)
        return attention_weighted_encoding, alpha
    
class DecoderAttention(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim):
        super(DecoderAttention, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Khởi tạo LSTM Cell (xử lý từng bước thời gian một thay vì cả chuỗi)
        # Đầu vào của cell = vector của từ + context vector từ hình ảnh
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        
        # Layer khởi tạo trạng thái ban đầu (h0, c0) cho LSTM dựa vào ảnh
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def forward(self, encoder_out, encoded_captions):
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size
        seq_length = encoded_captions.size(1)

        # Khởi tạo h0, c0 bằng cách trung bình cộng 49 pixel của hình ảnh
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        embeddings = self.embedding(encoded_captions) # (batch, seq_length, embed_size)
        
        # Khởi tạo ma trận chứa dự đoán của tất cả các bước thời gian
        predictions = torch.zeros(batch_size, seq_length, vocab_size).to(encoder_out.device)

        # Lặp qua từng thời điểm t để sinh từ (Time-step logic)
        for t in range(seq_length):
            # Tính Context Vector (vùng ảnh quan trọng nhất tại thời điểm t)
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            
            # Gộp Embedding của từ hiện tại với Context Vector
            lstm_input = torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1)
            
            # Đưa vào LSTMCell để cập nhật trạng thái h, c
            h, c = self.decode_step(lstm_input, (h, c))
            
            # Dự đoán xác suất ra từ mới
            preds = self.fc(h)
            predictions[:, t, :] = preds

        return predictions
    
    def generate_caption(self, encoder_out, max_length=20, start_idx=1, end_idx=2):
        # --- DÙNG KHI ĐÁNH GIÁ (INFERENCE) ---
        captions = []
        
        # Khởi tạo h0, c0 y như lúc train
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        
        # Bắt đầu với token <start>
        word_idx = torch.tensor([start_idx]).to(encoder_out.device)
        
        for _ in range(max_length):
            embeddings = self.embedding(word_idx) # (1, embed_size)
            
            # Tính Context Vector
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            
            # Nối và đưa vào LSTM Cell
            lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))
            
            # Dự đoán
            preds = self.fc(h)
            predicted = preds.argmax(1)
            captions.append(predicted.item())
            
            if predicted.item() == end_idx:
                break
                
            word_idx = predicted
            
        return captions