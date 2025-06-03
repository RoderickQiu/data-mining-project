import pandas as pd
from recommend.config import *

# Load the datasets (please adjust the paths)
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_questions = pd.read_csv(QUESTIONS_DATA_PATH)
df_difficulty = pd.read_csv(DIFFICULTY_DATA_PATH)

# merge
df_questions = df_questions.merge(df_difficulty, left_on="question_id", right_on="content_id", how="left")
df_questions.drop(columns=["content_id"], inplace=True)
df_questions.set_index("question_id", inplace=True)

def parse_tags(tags):
    if isinstance(tags, str):
        return list(map(int, tags.split()))
    return []


df_questions['tags_list'] = df_questions['tags'].apply(parse_tags)

# benchmark_tags: List of tag IDs (integers) to prioritize.
def get_recommendations(user_id, benchmark_tags=None, num_recommend=10):
    # user_attempted_quesionts
    user_attempted = df_train[(df_train['user_id'] == user_id) & (df_train['content_type_id'] == 0)]['content_id'].unique()
    
    user_correct = df_train[(df_train['user_id'] == user_id) & (df_train['content_type_id'] == 0) & (df_train['answered_correctly'] == 1)]['content_id'].unique()
    
    user_wrong = df_train[(df_train['user_id'] == user_id) & (df_train['content_type_id'] == 0) & (df_train['answered_correctly'] == 0)]['content_id'].unique()
    
    # Extract correct tags
    correct_questions_df = df_questions[df_questions['question_id'].isin(user_correct)]
    correct_tags = set()
    for tags in correct_questions_df['tags_list']:
        correct_tags.update(tags)
        
    # Extract wrong tags
    wrong_questions_df = df_questions[df_questions['question_id'].isin(user_wrong)]
    wrong_tags = set()
    for tags in wrong_questions_df['tags_list']:
        wrong_tags.update(tags)
    
    if not correct_tags and not wrong_tags:
        return []
    
    benchmark_tags = set(benchmark_tags) if benchmark_tags is not None else set()
    
    all_questions = set(df_questions['question_id'])
    not_attempted = all_questions - set(user_attempted)
    
    candidates = pd.DataFrame({'question_id': list(not_attempted)})
    
    # Compute score
    def overlap_score(qid):
        q_tags = set(df_questions.loc[df_questions['question_id'] == qid, 'tags_list'].values[0])
        common_with_correct = len(q_tags & correct_tags)
        common_with_wrong = len(q_tags & wrong_tags)
        common_with_benchmark = len(q_tags & benchmark_tags)
        total_score = 1.5 * common_with_wrong + 2 * common_with_benchmark - common_with_correct
        return total_score
    
    candidates['score'] = candidates['question_id'].apply(overlap_score)

    # sorting candidates
    recommended = candidates.sort_values(by='score', ascending=False).head(num_recommend)['question_id'].tolist()
    
    return recommended

def get_recommendations_advanced(user_id, benchmark_tags=None, num_recommend=10):

    user_attempts = df_train[(df_train['user_id'] == user_id) & (df_train['content_type_id'] == 0)]
    if user_attempts.empty:
        return [] 
    
    num_tags = 188 
    C_t = [0] * num_tags
    I_t = [0] * num_tags  
    
    # Process each attempt (user score of each tag)
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
    
    # Compute mastery scores M_t
    M_t = [(1 + C) / (2 + C + I) for C, I in zip(C_t, I_t)]
    
    # Get unattempted questions
    user_attempted = user_attempts['content_id'].unique()
    all_questions = set(df_questions.index)
    not_attempted = all_questions - set(user_attempted)
    candidates = pd.DataFrame({'question_id': list(not_attempted)})
    
    B = set(benchmark_tags) if benchmark_tags is not None else set()
    
    # Compute scores for candidate questions
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
    
    candidates['score'] = candidates['question_id'].apply(compute_score)
    
    recommended = candidates.sort_values(by='score', ascending=False).head(num_recommend)['question_id'].tolist()
    return recommended


if __name__ == "__main__":
    import time
    start = time.time()
    recommendations = get_recommendations_advanced(user_id=115, benchmark_tags=[51, 131], num_recommend=10)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print(recommendations)