import axios from 'axios';

const API_BASE_URL = 'http://10.16.87.206:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface PredictRequest {
    input_ids: number[];
    input_rtime: number[];
    input_cat: number[];
}

export interface PredictResponse {
    probs: number[];
}

export interface RecommendRequest {
    user_id: number;
    benchmark_tags?: number[];
    num_recommend: number;
}

export interface RecommendResponse {
    question_ids: number[];
}

export const apiService = {
    predict: async (data: PredictRequest): Promise<PredictResponse> => {
        const response = await api.post('/predict', data);
        return response.data;
    },

    recommend: async (data: RecommendRequest): Promise<RecommendResponse> => {
        const response = await api.post('/recommend', data);
        return response.data;
    },

    recommendAdvanced: async (data: RecommendRequest): Promise<RecommendResponse> => {
        const response = await api.post('/recommend_advanced', data);
        return response.data;
    },

    healthCheck: async (): Promise<{ message: string }> => {
        const response = await api.get('/');
        return response.data;
    },
}; 