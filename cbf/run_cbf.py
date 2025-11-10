import torch
import numpy as np
from cbf.config import SEED
from cbf.data import load_train_data, load_user_data
from cbf.model import train_autoencoder_for_emotion
from cbf.utils import (
    build_user_pref_vectors,
    embed_all_tracks_per_emotion,
    recommend_all_users,
    visualize_tsne_per_emotion,
)

def run_cbf():
    """Content-Based Filtering (CBF) 파이프라인을 실행합니다."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] CBF device = {device}")

    # 1. train 데이터 로드 및 감정 매핑
    print("\n[CBF] 1. 데이터 로드 중...")
    df_train_raw, emotion_to_data, emotion_id2text, text2emotion_id = load_train_data()

    # 2. 감정별 Autoencoder 학습
    print("\n[CBF] 2. 감정별 Autoencoder 모델 학습 중...")
    emotion_to_model = {}
    for eid, pack in emotion_to_data.items():
        feats_np = pack["features"]
        print(f"[INFO] Training AE for emotion {eid} with {feats_np.shape[0]} tracks...")
        ae_model = train_autoencoder_for_emotion(
            emotion_id=eid,
            features_np=feats_np,
            device=device,
        )
        emotion_to_model[eid] = ae_model

    # 3. user 데이터 로드
    print("\n[CBF] 3. 사용자 데이터 로드 중...")
    user_df = load_user_data(text2emotion_id=text2emotion_id)

    # 4. 유저별 취향 벡터 산출
    print("\n[CBF] 4. 사용자별 취향 벡터 산출 중...")
    user_pref, user_hist = build_user_pref_vectors(
        user_df=user_df,
        emotion_to_model=emotion_to_model,
        device=device,
    )

    # 5. 전체 트랙 latent 임베딩
    print("\n[CBF] 5. 전체 트랙 latent 임베딩 중...")
    emotion_to_trackinfo = embed_all_tracks_per_emotion(
        emotion_to_data=emotion_to_data,
        emotion_to_model=emotion_to_model,
        device=device,
    )

    # 6. 추천 생성 및 CF용 데이터 저장
    print("\n[CBF] 6. 추천 생성 및 CF 전달용 데이터 저장 중...")
    recommend_all_users(
        user_pref=user_pref,
        user_hist=user_hist,
        emotion_to_trackinfo=emotion_to_trackinfo,
        df_train_raw=df_train_raw,
        emotion_id2text=emotion_id2text,
    )

    # 7. 감정별 latent space 시각화
    print("\n[CBF] 7. Latent space 시각화 저장 중...")
    visualize_tsne_per_emotion(
        emotion_to_trackinfo=emotion_to_trackinfo,
        emotion_id2text=emotion_id2text,
    )
    print("\n[CBF] Content-Based Filtering 완료.")

if __name__ == "__main__":
    run_cbf()
