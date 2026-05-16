from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from peft import LoraConfig, get_peft_model

class BlipCaptioner(torch.nn.Module):
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", use_lora=True):
        super(BlipCaptioner, self).__init__()
        # Processor đảm nhiệm việc tokenize text và resize/normalize ảnh theo chuẩn của BLIP
        # Nó gom cả tokenizer và image_processor vào 1 hàm gọi cho gọn
        self.processor = BlipProcessor.from_pretrained(model_name)
        
        # Load kiến trúc Transformer (Vision Transformer + Text Decoder) đã được train
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)

        if use_lora:
            print("[INFO] Đang kích hoạt LoRA PEFT cho BLIP...")
            # Cấu hình LoRA adapter (Chỉ train các lớp Attention)
            config=LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"], 
                lora_dropout=0.05,
                bias="none"
            )
            self.model = get_peft_model(self.model, config)

            # Đóng băng hoàn toàn Vision Encoder
            for param in self.model.base_model.model.vision_model.parameters():
                param.requires_grad = False

            # In ra thống kê tham số
            self.model.print_trainable_parameters()


    def forward(self, raw_images, raw_texts):

        # Processor sẽ tự động chuyển đổi ảnh và text thành pixel_values, input_ids và attention_mask
        inputs = self.processor(
            images=raw_images, 
            text=raw_texts, 
            return_tensors="pt", 
            padding=True
        )

        # Đẩy toàn bộ các tensor vừa sinh ra lên GPU (hoặc CPU)
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Truyền vào mô hình. Tham số labels sẽ kích hoạt tính năng tự tính Loss.
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        return outputs.loss
    
    def generate_caption(self, raw_images, max_length=50):
        # Dùng cho quá trình Đánh giá (Test/Inference)
        inputs = self.processor(images=raw_images, return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=max_length)
        captions = self.processor.batch_decode(out_ids, skip_special_tokens=True)
        return captions