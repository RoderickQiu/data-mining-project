import torch
import platform


class Config:
    # Force CPU on macOS to avoid MPS issues, allow CUDA on other systems
    if platform.system() == "Darwin":  # macOS
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MAX_SEQ = 100
    EMBED_DIMS = 512
    ENC_HEADS = DEC_HEADS = 8
    NUM_ENCODER = NUM_DECODER = 4
    BATCH_SIZE = 32
    TRAIN_FILE = "/Users/dvalab/Documents/Roderick/kaggle-riiid/train.csv"
    TOTAL_EXE = 13523
    TOTAL_CAT = 10000
