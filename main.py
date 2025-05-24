import torch
import numpy as np
from config import Config
from train import PlusSAINTModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(checkpoint_path):
    model = PlusSAINTModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.to(device).eval()
    return model


# 使用您提供的5个具体样例
def prepare_input_samples():
    samples = [
        # 样例1
        {
            "input_ids": [7900,  7876,   175,  1278 , 2064 , 2065 , 2063 , 3364 , 3363 , 3365 , 2948  ,2946],
            "input_rtime": [0,   0 , 22 , 22  ,27  ,17  ,17  ,17  ,26  ,26 , 26,  24  ],
            "input_cat": [0,  1 , 2 , 3 , 4 , 4  ,4  ,5 , 5 , 5 , 6 , 6]
        },
        # # 样例2
        # {
        #     "input_ids": [639, 4123, 6152, 5318, 6409, 4128, 148, 22, 161, 10654],
        #     "input_rtime": [0, 23, 18, 23, 21, 53, 33, 37, 21, 22],
        #     "input_cat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # },
        # # 样例3
        # {
        #     "input_ids": [4123, 6152, 5318, 6409, 4128, 148, 22, 161, 10654, 5],
        #     "input_rtime": [0, 18, 23, 21, 53, 33, 37, 21, 22, 25],
        #     "input_cat": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # },
        # # 样例4
        # {
        #     "input_ids": [6152, 5318, 6409, 4128, 148, 22, 161, 10654, 5, 64],
        #     "input_rtime": [0, 23, 21, 53, 33, 37, 21, 22, 25, 22],
        #     "input_cat": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # },
        # # 样例5
        # {
        #     "input_ids": [5318, 6409, 4128, 148, 22, 161, 10654, 5, 64, 7911],
        #     "input_rtime": [0, 21, 53, 33, 37, 21, 22, 25, 22, 22],
        #     "input_cat": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        # }
    ]

    processed_samples = []
    for sample in samples:
        max_seq = Config.MAX_SEQ
        seq_len = len(sample["input_ids"])

        # 处理padding
        padded_ids = np.zeros(max_seq, dtype=np.int64)
        padded_ids[-seq_len:] = sample["input_ids"]

        padded_time = np.zeros(max_seq, dtype=np.int64)
        padded_time[-seq_len:] = sample["input_rtime"]

        padded_cat = np.zeros(max_seq, dtype=np.int64)
        padded_cat[-seq_len:] = sample["input_cat"]

        processed_samples.append({
            "input_ids": torch.from_numpy(padded_ids).unsqueeze(0).to(device),
            "input_rtime": torch.from_numpy(padded_time).unsqueeze(0).to(device),
            "input_cat": torch.from_numpy(padded_cat).unsqueeze(0).to(device)
        })

    return processed_samples


def predict_next_question(model, input_data):
    with torch.no_grad():
        dummy_labels = torch.zeros_like(input_data["input_ids"]).to(device)
        logits = model(input_data, dummy_labels)

        # 维度保护
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # 返回所有位置的预测概率（而不仅是最后一个）
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs


if __name__ == "__main__":
    model = load_trained_model("saved_models/best_model-v3.ckpt")
    samples = prepare_input_samples()

    for i, sample in enumerate(samples, 1):
        probs = predict_next_question(model, sample)

        # 打印完整预测结果
        print(f"\n样例{i}预测结果:")
        print(f"  题目ID: {sample['input_ids'].cpu().numpy()[0][-10:]}")  # 显示最后10个
        print(f"  预测概率: {probs}")  # 对应最后10题的预测概率
        print(f"  答题时间: {sample['input_rtime'].cpu().numpy()[0][-10:]}")
        print(f"  题目类别: {sample['input_cat'].cpu().numpy()[0][-10:]}")