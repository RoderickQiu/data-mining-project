# About
This is repository for Data Mining(CS306) final project.

# Back-end
## Data Tracing
运行train开始训练
main会访问saved_models的模型，根据输入输出一个包含正确概率的数组，目前错误率是65%，反过来错误率理论上就到35%了，所以也可以说正确率是65%
git上传不了这个模型，要的话得找我
数据集在https://www.kaggle.com/competitions/riiid-test-answer-prediction/data

## Recommending System
Please addjust the dataset path in `backend/recommend.py`

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