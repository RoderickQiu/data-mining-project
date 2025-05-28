import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from config import Config
from saint import PlusSAINTModule
from sample_parser import parse_document_to_json
import re

CKPT_PATH = 'saved_models/best_model-v3.ckpt'
BATCH_SIZE = 1

class EvaluationDataset(Dataset):
    def __init__(self, samples, max_seq):
        self.samples = samples
        self.max_seq = max_seq
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        seq_len = len(s['input_ids'])
        # padding
        input_ids = np.zeros(self.max_seq, dtype=np.int64)
        input_rtime = np.zeros(self.max_seq, dtype=np.int64)
        input_cat = np.zeros(self.max_seq, dtype=np.int64)
        labels = np.zeros(self.max_seq, dtype=np.int64)
        input_ids[-seq_len:] = s['input_ids'][:self.max_seq]
        input_rtime[-seq_len:] = s['input_rtime'][:self.max_seq]
        input_cat[-seq_len:] = s['input_cat'][:self.max_seq]
        labels[-seq_len:] = s['labels'][:self.max_seq]
        input = {
            'input_ids': torch.from_numpy(input_ids),
            'input_rtime': torch.from_numpy(input_rtime),
            'input_cat': torch.from_numpy(input_cat)
        }
        return input, torch.from_numpy(labels)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input, labels = batch
            for k in input:
                input[k] = input[k].to(device)
            labels = labels.to(device)
            target_mask = (input['input_ids'] != 0)
            out = model(input, labels)
            out = torch.masked_select(out, target_mask)
            out = torch.sigmoid(out)
            labels = torch.masked_select(labels, target_mask)
            all_preds.append(out.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            # print(f"Batch predictions: {out.cpu().numpy()}")
            # print(f"Batch labels: {labels.cpu().numpy()}")
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    print(f"Eval AUC: {auc:.6f}")
    return auc

def main():
    device = torch.device("cpu")
    test_document = open('val_samples.txt', 'r').read()
    samples = parse_document_to_json(test_document)['samples']
    dataset = EvaluationDataset(samples, max_seq=Config.MAX_SEQ)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print('Loading model...')
    model = PlusSAINTModule()
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.to(device)
    print('Evaluating...')
    evaluate(model, dataloader, device)

if __name__ == '__main__':
    main() 