import json
import os
import warnings
from evaluate import load
from pycocoevalcap.cider.cider import Cider
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def compute_cider(predictions, references):

    """
    Hàm tính CIDEr dựa trên thư viện chuẩn pycocoevalcap.
    Yêu cầu định dạng dict: {id: ['câu 1', 'câu 2']}, thay vì list.
    """

    res = {i: [preds] for i, preds in enumerate(predictions)}
    gts = {i: refs for i, refs in enumerate(references)}
    
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return score

def evaluate_metrics(json_path, bleu_metric, meteor_metric):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = []
    references = []

    for item in data:
        predictions.append(item["prediction"])
        
        # Xử lý reference (Vì dataset Flickr30k có 5 ground truth cho 1 ảnh)
        refs = item["reference"]
        if isinstance(refs, str):
            refs = [refs]
        references.append(refs)

    # Tính BLEU
    bleu_scores = bleu_metric.compute(predictions=predictions, references=references)
    
    # Tính METEOR
    meteor_scores = meteor_metric.compute(predictions=predictions, references=references)
    
    # Tính CIDEr
    cider_score = compute_cider(predictions, references)

    return {
        "BLEU-1": bleu_scores["precisions"][0],
        "BLEU-4": bleu_scores["precisions"][3],
        "METEOR": meteor_scores["meteor"],
        "CIDEr": cider_score
    }

def save_table_as_image(table_data, save_path):
    """
    Hàm vẽ và lưu bảng kết quả thành file ảnh PNG nét cao.
    """
    fig, ax = plt.subplots(figsize=(10, 4)) # Tùy chỉnh kích thước ảnh
    ax.axis('tight')
    ax.axis('off')
    
    # Vẽ bảng
    table = ax.table(
        cellText=table_data,
        colLabels=None, # Đã gộp header vào table_data
        loc='center',
        cellLoc='center'
    )
    
    # Làm đẹp bảng
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8) # Kéo giãn khoảng cách các ô cho dễ nhìn
    
    # Tô màu cho hàng tiêu đề (Header)
    for j in range(len(table_data[0])):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4c72b0') # Màu xanh biển đậm
        
    # Bôi đậm dòng chứa kết quả của BLIP (Dòng cuối cùng)
    for j in range(len(table_data[0])):
        cell = table[len(table_data)-1, j]
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e8f4f8') # Màu xanh lơ nhạt làm nổi bật

    plt.title("Bảng so sánh hiệu năng các mô hình Image Captioning", fontweight="bold", fontsize=14, pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # dpi=300 để ảnh cực nét khi in ra giấy
    plt.close()
    print(f"\n[INFO] Đã xuất bảng kết quả thành ảnh tại: {save_path}")

def main():
    print("[INFO] Đang tải các mô-đun đánh giá NLP (BLEU, METEOR, CIDEr)...")
    bleu = load("bleu")
    meteor = load("meteor")

    models_to_eval = {
        "CNN-LSTM      ": "results/CNN_LSTM_results.json",
        "Attention     ": "results/ATTENTION_results.json",
        "BLIP + LoRA   ": "results/BLIP_LoRA_results.json"
    }

    # Header của bảng
    table_data = [["MÔ HÌNH", "BLEU-1", "BLEU-4", "METEOR", "CIDEr"]]

    print("\n" + "="*65)
    print(f"{'MÔ HÌNH':<16} | {'BLEU-1':<8} | {'BLEU-4':<8} | {'METEOR':<8} | {'CIDEr':<8}")
    print("-" * 65)

    for model_name, path in models_to_eval.items():
        if not os.path.exists(path):
            print(f"{model_name:<16} | --- CHƯA CÓ DỮ LIỆU ---")
            continue
            
        scores = evaluate_metrics(path, bleu, meteor)
        
        b1 = f"{scores['BLEU-1']:.4f}"
        b4 = f"{scores['BLEU-4']:.4f}"
        met = f"{scores['METEOR']:.4f}"
        cid = f"{scores['CIDEr']:.4f}" # CIDEr thường có giá trị từ 0.0 đến ~2.0

        table_data.append([model_name, b1, b4, met, cid])
        
        print(f"{model_name:<16} | {b1:<8} | {b4:<8} | {met:<8} | {cid:<8}")
        
    print("="*65)

    # Gọi hàm xuất ảnh
    os.makedirs("results", exist_ok=True)
    save_table_as_image(table_data, "results/evaluation_table.png")

if __name__ == "__main__":
    main()