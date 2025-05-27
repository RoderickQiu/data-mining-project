import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    MAX_SEQ = 100
    EMBED_DIMS = 512
    ENC_HEADS = DEC_HEADS = 8
    NUM_ENCODER = NUM_DECODER = 4
    BATCH_SIZE = 32
    TRAIN_FILE = "/Users/dvalab/Documents/Roderick/kaggle-riiid/train.csv"
    TOTAL_EXE = 13523
    TOTAL_CAT = 10000
