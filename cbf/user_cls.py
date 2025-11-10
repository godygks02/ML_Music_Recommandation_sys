import numpy as np
import pandas as pd
import torch

from .config import (
    USER_CSV,
    FEATURE_COLS,
    USER_ID_COL,
    USER_EMOTION_TEXT_COL,
    USER_TRACK_ID_COL,
)
from .utils import l2_normalize_np

def load_user_dataset_scaled(emotion_text2id, robust, minmax):
    """
    기존 버전은 robust, minmax로 유저 피처를 train 스케일에 맞춰 변환했었음.
    이제 전처리 스케일링을 쓰지 않으므로,
    그냥 raw FEATURE_COLS 값(float32) 그대로 쓴다.

    반환:
      df_u          : 유저 데이터 (raw feature 포함)
      scaled_cols   : 모델 입력에 쓸 컬럼 이름 리스트 (여전히 scaled_* 이름으로 맞춰줌)
                      -> downstream 코드가 scaled_*을 참조하므로 형태만 유지
    """

    df_u = pd.read_csv(USER_CSV)

    # user_id 그대로 (문자/숫자 상관 없이 string으로 통일하거나 유지)
    df_u["user_id"] = df_u[USER_ID_COL].astype(str)

    # 감정 문자열 -> emotion_id 정수
    df_u["emotion_id"] = df_u[USER_EMOTION_TEXT_COL].map(emotion_text2id).astype(int)

    # 이제는 스케일링 안 함: raw feature 그대로 사용
    Xu_raw = df_u[FEATURE_COLS].astype(np.float32).values  # (N_user, d)

    # downstream 코드가 scaled_컬럼을 기대하고 있으니까 그대로 만든다.
    scaled_cols = []
    for col_idx, col in enumerate(FEATURE_COLS):
        sc_col = f"scaled_{col}"
        df_u[sc_col] = Xu_raw[:, col_idx]
        scaled_cols.append(sc_col)

    return df_u, scaled_cols


@torch.no_grad()
def build_user_pref_vectors(df_user_scaled, scaled_cols, model, device):
    """
    유저별(user_id) × 감정별(emotion_id):
    - 그 유저가 그 감정일 때 들은 곡들의 feature들을 모델에 넣어서 임베딩(fc2)을 구하고
    - 평균낸 후 L2 normalize한 벡터를 user_pref_vecs[user_id][emotion_id]로 저장.
    - 그 감정에서 이미 들은 track_id 목록은 user_history에 저장.

    이 부분은 정규화(스케일링)과는 무관하므로 기존 로직 유지.
    """

    model.eval()

    user_pref_vecs = {}
    user_history   = {}

    for user_id, df_user_grp in df_user_scaled.groupby("user_id"):
        emo2vec  = {}
        emo2hist = {}

        for emo_id, df_user_emo in df_user_grp.groupby("emotion_id"):
            feats_user = df_user_emo[scaled_cols].values.astype(np.float32)
            feats_t = torch.tensor(feats_user, dtype=torch.float32).to(device)

            emb_t = model(feats_t, return_emb=True)  # (m, EMB_DIM)
            emb_mean = emb_t.mean(dim=0).cpu().numpy().astype(np.float32)
            emb_mean = l2_normalize_np(emb_mean)

            emo2vec[int(emo_id)] = emb_mean

            listened_ids = df_user_emo[USER_TRACK_ID_COL].astype(str).tolist()
            emo2hist[int(emo_id)] = listened_ids

        user_pref_vecs[user_id] = emo2vec
        user_history[user_id]   = emo2hist

    return user_pref_vecs, user_history
