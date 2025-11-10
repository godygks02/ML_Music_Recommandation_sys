import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from .config import (
    TRAIN_CSV,
    USER_CSV,
    FEATURE_COLS,
    EMOTION_COL,              # train 쪽 감정 컬럼명: "new_label"
    TRACK_ID_TRAIN_COL,       # train 곡 ID: "uri"
    TRACK_ID_USER_COL,        # user 곡 ID: "track_id"
    USER_ID_COL,
    BATCH_SIZE,
    RANDOM_STATE,
)


class FeatureOnlyDataset(Dataset):
    """
    Autoencoder용 Dataset: 입력 x 자체를 복원 타겟으로 사용.
    """
    def __init__(self, feats: np.ndarray):
        self.X = feats.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        return x, x  # AE: 입력 == 타겟


def load_train_data():
    """
    train_labeled_dataset.csv 로드.
    - EMOTION_COL(new_label)을 기준으로 emotion_id(정수) 생성.
    - emotion_id별로 features / uri 묶어서 딕셔너리화.
    - emotion_id2text (정수ID -> 라벨텍스트)
      text2emotion_id (라벨텍스트 -> 정수ID) 생성.
    반환:
      df_train_raw, emotion_to_data, emotion_id2text, text2emotion_id
    """
    df = pd.read_csv(TRAIN_CSV)

    # 감정 라벨을 정수 ID로 변환
    if df[EMOTION_COL].dtype == object:
        le = LabelEncoder()
        df["emotion_id"] = le.fit_transform(df[EMOTION_COL].astype(str))
        emotion_id2text = {i: t for i, t in enumerate(le.classes_)}
        text2emotion_id = {t: i for i, t in enumerate(le.classes_)}
    else:
        df["emotion_id"] = df[EMOTION_COL].astype(int)
        uniq = sorted(df["emotion_id"].unique())
        emotion_id2text = {eid: f"Emotion {eid}" for eid in uniq}
        text2emotion_id = {f"Emotion {eid}": eid for eid in uniq}

    # 곡 스타일 feature 행렬
    feat_mat = df[FEATURE_COLS].astype(np.float32).values  # (N, D)

    # 곡 식별자 (train 기준: uri)
    track_ids = df[TRACK_ID_TRAIN_COL].astype(str).values  # (N,)

    # 감정 ID
    emotion_ids = df["emotion_id"].astype(int).values      # (N,)

    # 감정별로 묶은 dict
    emotion_to_data = {}
    for eid in sorted(np.unique(emotion_ids)):
        mask = (emotion_ids == eid)
        emotion_to_data[eid] = {
            "features": feat_mat[mask],        # (n_eid, D)
            "track_ids": track_ids[mask],      # (n_eid,)
        }

    return df, emotion_to_data, emotion_id2text, text2emotion_id


def _pick_user_emotion_col(user_df):
    """
    user_dataset.csv 내부에서 감정을 나타내는 컬럼명을 자동으로 찾는다.
    우선순위대로 첫 번째로 존재하는 컬럼을 쓴다.
    new_label이 없을 수도 있으니까 fallback을 준다.
    """
    candidate_cols = [
        "new_label",        # 우리가 기대한 최종 감정 라벨
        "labels",           # 흔히 있는 감정/분류 라벨 이름
        "label",
        "predicted_label",  # 예측된 감정 라벨
        "emotion",
        "emotion_label",
        "class_cluster",    # 혹시 클러스터 id 텍스트화된 감정 라벨일 수도 있음
    ]
    for c in candidate_cols:
        if c in user_df.columns:
            return c
    raise RuntimeError(
        f"user_dataset에서 감정 라벨 컬럼을 찾을 수 없습니다. "
        f"columns={list(user_df.columns)}"
    )


def load_user_data(text2emotion_id):
    """
    user_dataset.csv 로드.
    - user_id, track_id(user 기준 곡ID), 감정 라벨 -> emotion_id 로 정수화.
    반환: uf (DataFrame)
      필수 컬럼:
        user_id (str)
        track_id (str)  <- user 쪽 곡 id
        emotion_id (int)
        + FEATURE_COLS 전부
    """
    uf = pd.read_csv(USER_CSV)

    # user_id와 track_id 정리
    uf["user_id"] = uf[USER_ID_COL].astype(str)
    uf["track_id"] = uf[TRACK_ID_USER_COL].astype(str)

    # user_dataset 안에서 실제 감정 라벨이 담긴 컬럼명을 자동으로 고른다.
    user_emotion_col = _pick_user_emotion_col(uf)

    # case1: user 쪽 감정 라벨이 문자열이고 train의 text2emotion_id에 매핑 가능
    # case2: 매핑 안 되면 fallback으로 factorize
    if uf[user_emotion_col].dtype == object:
        if text2emotion_id is not None:
            # train 기준으로 본 감정 텍스트 -> emotion_id 매핑 시도
            uf["emotion_id"] = uf[user_emotion_col].map(text2emotion_id)
        else:
            uf["emotion_id"] = None

        # 매핑 실패한 값(NaN)이 있으면 그 부분만 factorize fallback
        if uf["emotion_id"].isna().any():
            # factorize로 새로 만든다 (이건 train과 불일치 가능성 있음)
            # 그래도 코드가 죽지는 않게 하기 위한 최후 수단
            valid_mask = uf["emotion_id"].isna()
            # factorize 전체 돌리면 감정 전체가 다시 재라벨링되니까,
            # 그냥 통일성 위해 전부 factorize로 다시 만드는 게 낫다.
            codes, _ = pd.factorize(uf[user_emotion_col].astype(str))
            uf["emotion_id"] = codes.astype(int)
        else:
            uf["emotion_id"] = uf["emotion_id"].astype(int)

    else:
        # 감정 라벨이 이미 숫자일 때
        uf["emotion_id"] = uf[user_emotion_col].astype(int)

    return uf


def build_ae_loader_for_array(features_np):
    """
    감정 하나에 해당하는 feature array (N, D)를 받아
    Autoencoder 학습용 DataLoader를 돌려준다.
    """
    ds = FeatureOnlyDataset(features_np)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    return dl
