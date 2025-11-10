# config.py

# 파일 경로
TRAIN_CSV = "data/train_labeled_dataset.csv"
USER_CSV = "data/user_dataset.csv"

# 출력물 경로
CBF_OUTPUT_DIR = "output/cbf_intermediate"
OUTPUT_TSNE_PNG = "output/embedding_tsne.png"

# 오디오/스타일 특성으로 쓸 수치형 feature들
FEATURE_COLS = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# 감정(최종 라벨)
EMOTION_COL = "new_label"

# 곡 ID 컬럼 이름 (train / user에서 다르다)
TRACK_ID_TRAIN_COL = "uri"        # train_labeled_dataset.csv 기준
TRACK_ID_USER_COL = "track_id"    # user_dataset.csv 기준

# 유저 ID 컬럼
USER_ID_COL = "user_id"

# Autoencoder 학습 설정
EPOCHS_AE = 10
BATCH_SIZE = 256
LR = 1e-3
STYLE_DIM = 32          # latent bottleneck 차원 (스타일 임베딩 차원)
VAL_RATIO = 0.2
RANDOM_STATE = 42

# 추천 관련
TOP_K_PER_EMOTION = 100
TSNE_PER_EMOTION_SAMPLE = 2000  # tsne 시 감정당 샘플 제한

SEED = 42
