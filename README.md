# About
This is repository for CS306 (Data Mining) final project.

# Back-end

## FastAPI

Now the `main.py` is a FastAPI server that can be used to predict the next question and recommend questions.

Run `uvicorn main:app --reload` to start the server. See the content of `main.py` for more details.

## Data Tracing
Run `train.py` to start training.
The `test.py` accesses the model in `saved_models` directory and outputs an array containing probabilities of correct answers. Currently the error rate is 65%, which theoretically means the accuracy rate is 35% when reversed, so we can also say the accuracy rate is 65%.
The model is too large for git and can be retrieved from contributors.
Dataset can be found at https://www.kaggle.com/competitions/riiid-test-answer-prediction/data

## Recommending System
Please addjust the dataset path in `backend/config.py`.

### Function

1. `get_recommendations`
```python
total_score = 1.5 * common_with_wrong + 2 * common_with_benchmark - common_with_correct
```
2. `get_recommendations_advanced`
```python
def compute_score(q):
    T_q = df_questions.loc[q, 'tags_list']
    sum_weak = sum(1 - M_t[t] for t in T_q if t < len(M_t))
    # add question difficulty
    D_q = df_questions.loc[q, 'difficulty']
    # normalization, that is to make sure for weaker tags' mastery score, we should recommend to relatively easier questions
    score = sum_weak / (1 + D_q) if (1 + D_q) > 0 else 0
    # if it is recommended question, we should add a bonus
    if df_questions.loc[q, 'quality_flag'] == "推荐":
        score *= 1.2
    # Add bonus for benchmark_tags
    score += 2 * len(set(T_q) & B)
    return score
```