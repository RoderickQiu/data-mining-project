# Intelligent Learning Recommendation System

This repository contains the codebase for a CS306 (Data Mining) final project, which implements an intelligent learning recommendation system. The system features a deep learning-based knowledge tracing model and a personalized question recommendation engine, with a modern web frontend for user interaction.

## Project Structure

```
data-mining-project/
├── recommend/              # Standalone recommendation logic (legacy, see main.py for API)
├── frontend/               # Modern React + TypeScript web interface
├── logs/                   # Training and experiment logs
├── saved_models/           # Trained model checkpoints (not tracked by git)
├── main.py                 # FastAPI backend server (primary API)
├── train.py                # Model training script
├── test.py                 # Model evaluation script
├── dataset.py              # Data loading utilities
├── requirements.txt        # Python dependencies
└── ...
```

---

## Features

### 1. Knowledge Tracing & Prediction

- Implements a deep learning model (SAINT+) for sequential knowledge tracing.
- Predicts the probability of a student answering the next question correctly, based on their historical responses, response times, and question categories.
- Model is built with PyTorch for modularity and scalability.

### 2. Personalized Question Recommendation

- Question Recommendation: Suggests questions based on user mastery of knowledge tags, question difficulty, quality flags, tag overlap with user strengths/weaknesses, and benchmark tags.
- Available via API endpoint `/recommend_advanced`.

### 3. Modern Web Frontend

- Built with React, TypeScript, Tailwind CSS, and shadcn/ui.
- Provides interactive prediction and recommendation interfaces.
- Real-time data visualization with Recharts.
- Responsive design for desktop and mobile.

---

## Getting Started

### Backend (FastAPI)

#### Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

#### Running the API Server

1. Ensure the trained model checkpoint is available at `saved_models/best_model-v3.ckpt`.
2. Adjust dataset paths in `main.py` or `backend/config.py` as needed.
3. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
4. The API will be available at `http://localhost:8000`.

#### Key API Endpoints

- `POST /predict`: Predicts the probability of correct answers for a sequence.
- `POST /recommend_advanced`: Returns advanced personalized question recommendations.
- `POST /question_stats_by_ids`: Retrieves metadata for a list of question IDs.

### Model Training

- To train the model from scratch:
  ```bash
  python train.py
  ```
- Training logs and metrics are saved in the `logs/` directory.

### Frontend

#### Setup & Run

```bash
cd frontend
npm install
npm start
```

- The app will be available at `http://localhost:3000`.
- API base URL is configured in `src/services/api.ts` (default: `http://localhost:8000`).

#### Features

- **Learning Prediction**: Input question IDs, response times, and categories to visualize predicted probabilities.
- **Question Recommendation**: Get personalized question suggestions using basic or advanced algorithms.
- **Data Visualization**: Interactive charts for predictions and recommendations.

#### Project Structure (Frontend)

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/        # React components
│   │   ├── ui/            # shadcn/ui base components
│   │   ├── MainTab.tsx    # Main tab
│   ├── services/          # API services
│   │   └── api.ts         # API interface
│   ├── lib/               # Utility functions
│   │   └── utils.ts       # General utilities
│   ├── App.tsx            # Main app component
│   └── index.tsx          # App entry point
├── package.json           # Project config
└── tailwind.config.js     # Tailwind config
```

---

## Datasets

- The system uses the [Riiid! Answer Prediction](https://www.kaggle.com/competitions/riiid-test-answer-prediction/data) dataset.
- Please download the dataset and adjust paths in `main.py` or `backend/config.py` as needed.

---

## Python Dependencies

- torch
- pytorch-lightning
- pandas
- numpy
- scikit-learn
- fastapi
- uvicorn

As declared in `requirements.txt`.

---

## Logs & Model Checkpoints

- Training logs are stored in `logs/`.
- Model checkpoints are saved in `saved_models/` (not tracked by git due to size).