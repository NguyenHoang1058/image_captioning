# 🖼️ Image Captioning: Từ Mạng Nơ-ron Cơ Sở Đến Transformer

Đây là mã nguồn của đồ án môn học Machine Learning. Project này triển khai và so sánh 3 kiến trúc học sâu khác nhau cho bài toán tự động sinh câu mô tả hình ảnh (Image Captioning).

## 🚀 Các mô hình được hỗ trợ
1. **CNN - LSTM (Baseline):** Trích xuất đặc trưng bằng ResNet50 và giải mã bằng chuỗi LSTM.
2. **Attention Model:** Nâng cấp với cơ chế chú ý cục bộ (Bahdanau Attention) giúp mô hình tập trung vào từng vùng ảnh cụ thể.
3. **BLIP + LoRA (SOTA):** Tinh chỉnh (Fine-tuning) mô hình ngôn ngữ lớn đa phương thức BLIP của Salesforce bằng kỹ thuật PEFT/LoRA.

---

## 🛠️ Hướng dẫn cài đặt

**Bước 1: Clone mã nguồn về máy**
```bash
# git clone [https://github.com/Tên-Github-Của-Bạn/image_captioning.git](https://github.com/Tên-Github-Của-Bạn/image_captioning.git)
# cd image_captioning

**Bước 2: Cài môi trường**
# Dùng terminal để chạy đoạn lệnh bên dưới
# python -m venv venv
# source venv\Scripts\activate  # Trong trường lỗi thì chạy lệnh sau : venv\Scripts\Activate.ps1
# pip install -r requirements.txt

**Bước 3: Demo**
# Để chạy demo thì thực thi lệnh sau ở terminal
# python predict_blip.py
# python predict_attention.py
# python predict_cnn_lstm.py
# Sau đó dùng chuột kéo ảnh từ thư mục demo (demo_images) rồi chờ kết quả

# Make by nguyeen