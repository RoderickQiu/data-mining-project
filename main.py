from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Body
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import pandas as pd
from config import Config
from train import PlusSAINTModule
import os

# For recommend.py data paths
RECOMMEND_DATA_PATHS = {
    'train': '/Users/dvalab/Documents/Roderick/kaggle-riiid/train.csv',
    'questions': '/Users/dvalab/Documents/Roderick/kaggle-riiid/questions.csv',
    'difficulty': '/Users/dvalab/Documents/Roderick/kaggle-riiid/question_difficulty_discrimination.csv',
}

def parse_tags(tags):
    if isinstance(tags, str):
        return list(map(int, tags.split()))
    return []

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

device = Config.device
model = None
# DataFrames for recommend endpoints
df_train = None
df_questions = None

class PredictRequest(BaseModel):
    input_ids: List[int]
    input_rtime: List[int]
    input_cat: List[int]

class PredictResponse(BaseModel):
    probs: List[float]

class RecommendRequest(BaseModel):
    user_id: int
    benchmark_tags: Optional[List[int]] = None
    num_recommend: int = 10

class RecommendResponse(BaseModel):
    question_ids: List[int]

class QuestionStatsResponse(BaseModel):
    question_id: int
    difficulty: float
    discrimination: float
    quality_flag: Optional[str] = None
    bundle_id: Optional[int] = None
    correct_answer: Optional[int] = None
    part: Optional[int] = None
    tags: Optional[str] = None

@app.on_event("startup")
def load_resources():
    global model, df_train, df_questions
    # Load model
    try:
        model = PlusSAINTModule.load_from_checkpoint(
            "saved_models/best_model-v3.ckpt",
            dtype=torch.float32
        )
        model.to(device).eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
    # Load recommend.py data
    try:
        df_train = pd.read_csv(RECOMMEND_DATA_PATHS['train'])
        df_questions = pd.read_csv(RECOMMEND_DATA_PATHS['questions'])
        df_difficulty = pd.read_csv(RECOMMEND_DATA_PATHS['difficulty'])
        df_questions_merged = df_questions.merge(df_difficulty, left_on="question_id", right_on="content_id", how="left")
        df_questions_merged.drop(columns=["content_id"], inplace=True)
        df_questions_merged.set_index("question_id", inplace=True)
        df_questions_merged['tags_list'] = df_questions_merged['tags'].apply(parse_tags)
        df_questions = df_questions_merged
    except Exception as e:
        print(f"Failed to load recommend data: {e}")
        df_train = None
        df_questions = None
        
@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    seq_len = len(request.input_ids)
    if not (len(request.input_rtime) == seq_len and len(request.input_cat) == seq_len):
        raise HTTPException(status_code=400, detail="All input lists must have the same length.")
    max_seq = Config.MAX_SEQ
    padded_ids = np.zeros(max_seq, dtype=np.int64)
    padded_ids[-seq_len:] = request.input_ids
    padded_time = np.zeros(max_seq, dtype=np.int64)
    padded_time[-seq_len:] = request.input_rtime
    padded_cat = np.zeros(max_seq, dtype=np.int64)
    padded_cat[-seq_len:] = request.input_cat
    input_data = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long).unsqueeze(0).to(device),
        "input_rtime": torch.tensor(padded_time, dtype=torch.long).unsqueeze(0).to(device),
        "input_cat": torch.tensor(padded_cat, dtype=torch.long).unsqueeze(0).to(device)
    }
    with torch.no_grad():
        dummy_labels = torch.zeros_like(input_data["input_ids"], dtype=torch.long).to(device)
        logits = model(input_data, dummy_labels)
        # mask 非padding部分
        target_mask = (input_data["input_ids"] != 0)
        logits = torch.masked_select(logits, target_mask)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
    return PredictResponse(probs=probs)

@app.post("/recommend_advanced", response_model=RecommendResponse)
def recommend_advanced(request: RecommendRequest):
    if df_train is None or df_questions is None:
        raise HTTPException(status_code=503, detail="Recommend data not loaded.")
    user_id = request.user_id
    benchmark_tags = set(request.benchmark_tags) if request.benchmark_tags else set()
    num_recommend = request.num_recommend
    user_attempts = df_train[(df_train['user_id'] == user_id) & (df_train['content_type_id'] == 0)]
    if user_attempts.empty:
        return RecommendResponse(question_ids=[])
    num_tags = 188
    C_t = [0] * num_tags
    I_t = [0] * num_tags
    # 统计正答/错答过的题目标签
    user_correct = user_attempts[user_attempts['answered_correctly'] == 1]['content_id'].unique()
    user_wrong = user_attempts[user_attempts['answered_correctly'] == 0]['content_id'].unique()
    correct_questions_df = df_questions[df_questions.index.isin(user_correct)]
    correct_tags = set()
    for tags in correct_questions_df['tags_list']:
        correct_tags.update(tags)
    wrong_questions_df = df_questions[df_questions.index.isin(user_wrong)]
    wrong_tags = set()
    for tags in wrong_questions_df['tags_list']:
        wrong_tags.update(tags)
    # 统计标签掌握度
    for _, row in user_attempts.iterrows():
        content_id = row['content_id']
        answered_correctly = row['answered_correctly']
        if content_id in df_questions.index:
            tags_list = df_questions.loc[content_id, 'tags_list']
            for t in tags_list:
                if answered_correctly == 1:
                    C_t[t] += 1
                elif answered_correctly == 0:
                    I_t[t] += 1
    M_t = [(1 + C) / (2 + C + I) for C, I in zip(C_t, I_t)]
    user_attempted = user_attempts['content_id'].unique()
    all_questions = set(df_questions.index)
    not_attempted = all_questions - set(user_attempted)
    candidates = pd.DataFrame({'question_id': list(not_attempted)})
    B = benchmark_tags
    def compute_score(q):
        T_q = df_questions.loc[q, 'tags_list']
        sum_weak = sum(1 - M_t[t] for t in T_q if t < len(M_t))
        D_q = df_questions.loc[q, 'difficulty']
        discrimination = df_questions.loc[q, 'discrimination'] if 'discrimination' in df_questions.columns else None
        # 剔除区分度过低的题目
        if discrimination is not None and discrimination < 0.2:
            return float('-inf')
        score = sum_weak / (1 + D_q) if (1 + D_q) > 0 else 0
        if 'quality_flag' in df_questions.columns and df_questions.loc[q].get('quality_flag', None) == "推荐":
            score *= 1.2
        # 融合正答/错答/benchmark标签思想
        q_tags = set(T_q)
        common_with_correct = len(q_tags & correct_tags)
        common_with_wrong = len(q_tags & wrong_tags)
        common_with_benchmark = len(q_tags & B)
        score += 1.5 * common_with_wrong + 2 * common_with_benchmark - common_with_correct
        return score
    candidates['score'] = candidates['question_id'].apply(compute_score)
    recommended = candidates.sort_values(by='score', ascending=False).head(num_recommend)['question_id'].tolist()
    return RecommendResponse(question_ids=recommended)

@app.post("/question_stats_by_ids")
async def question_stats_by_ids(question_ids: List[int] = Body(...)):
    if df_questions is None:
        raise HTTPException(status_code=503, detail="Recommend data not loaded.")
    result = []
    for qid in question_ids:
        if qid in df_questions.index:
            row = df_questions.loc[qid]
            entry = {
                'question_id': int(qid),
                'difficulty': float(row['difficulty']) if not pd.isna(row['difficulty']) else None,
                'discrimination': float(row['discrimination']) if not pd.isna(row['discrimination']) else None,
                'quality_flag': row['quality_flag'] if 'quality_flag' in row and not pd.isna(row['quality_flag']) else None,
                'bundle_id': int(row['bundle_id']) if 'bundle_id' in row and not pd.isna(row['bundle_id']) else None,
                'correct_answer': int(row['correct_answer']) if 'correct_answer' in row and not pd.isna(row['correct_answer']) else None,
                'part': int(row['part']) if 'part' in row and not pd.isna(row['part']) else None,
                'tags': row['tags'] if 'tags' in row and not pd.isna(row['tags']) else None
            }
            result.append(entry)
    return result 