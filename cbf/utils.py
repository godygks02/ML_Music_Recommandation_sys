import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from cbf.config import (
    TOP_K_PER_EMOTION,
    CBF_OUTPUT_DIR,
    STYLE_DIM,
    TSNE_PER_EMOTION_SAMPLE,
    FEATURE_COLS,
    TRACK_ID_TRAIN_COL,
)

# -----------------------
# 기본 유틸
# -----------------------

def l2_normalize_np(vec):
    n = np.linalg.norm(vec) + 1e-12
    return vec / n

def cosine_sim(a, b):
    return float(
        np.dot(a, b)
        / ((np.linalg.norm(a)+1e-12) * (np.linalg.norm(b)+1e-12))
    )

# -----------------------
# 유저 취향 벡터 계산
# -----------------------

def build_user_pref_vectors(user_df, emotion_to_model, device):
    """
    user_df: user_dataset (user_id, track_id, emotion_id, FEATURE_COLS...)
    emotion_to_model: {emotion_id: trained AE model}

    반환:
      user_pref[user_id][emotion_id] = latent mean (L2-normalized)
      user_hist[user_id][emotion_id] = 사용자가 가진 track_id 리스트 (user 기준 track_id)
    """
    user_pref = {}
    user_hist = {}

    for uid, df_user_u in user_df.groupby("user_id"):
        emo2vec = {}
        emo2hist = {}

        for eid, df_user_e in df_user_u.groupby("emotion_id"):
            if eid not in emotion_to_model:
                continue

            model = emotion_to_model[eid]

            feats_np = df_user_e[FEATURE_COLS].astype(np.float32).values
            feats_t = torch.tensor(feats_np, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, z = model(feats_t)  # (m, STYLE_DIM)
                z_mean = z.mean(dim=0).cpu().numpy().astype(np.float32)

            emo2vec[int(eid)] = l2_normalize_np(z_mean)

            listened_tracks_this_emotion = df_user_e["track_id"].astype(str).tolist()
            emo2hist[int(eid)] = listened_tracks_this_emotion

        user_pref[uid] = emo2vec
        user_hist[uid] = emo2hist

    return user_pref, user_hist

# -----------------------
# 전체 트랙 latent 추출
# -----------------------

def embed_all_tracks_per_emotion(emotion_to_data, emotion_to_model, device):
    """
    emotion_to_data[eid] = {
        "features": (N_eid, D),
        "track_ids": (N_eid,)
    }

    반환:
      emotion_to_trackinfo[eid] = {
        "track_ids": np.array([...], dtype=str),   # train 기준 track id (uri)
        "latents": np.ndarray [N_eid, STYLE_DIM],  # AE latent
    }
    """
    emotion_to_trackinfo = {}

    for eid, pack in emotion_to_data.items():
        if eid not in emotion_to_model:
            continue
        model = emotion_to_model[eid]

        feats_np = pack["features"].astype(np.float32)
        feats_t = torch.tensor(feats_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, z = model(feats_t)
            z_np = z.cpu().numpy().astype(np.float32)

        emotion_to_trackinfo[eid] = {
            "track_ids": np.array(pack["track_ids"], dtype=str),
            "latents": z_np,
        }

    return emotion_to_trackinfo

# -----------------------
# MMR (Maximal Marginal Relevance) 리랭킹
# -----------------------

def mmr_rerank(user_vec, cand_latents, cand_ids, top_k, lambda_div=0.7):
    """
    user_vec: (STYLE_DIM,) np.float32, L2-normalized된 유저 취향 벡터
    cand_latents: (N, STYLE_DIM) np.float32, 각 후보 곡 latent
    cand_ids: (N,) 곡 ID들 (uri)
    top_k: 최종적으로 뽑을 개수
    lambda_div: 0~1 사이. 1에 가까울수록 '유사도 위주', 낮을수록 '다양성 위주'.

    반환:
      sel_ids, sel_scores, sel_idx
      sel_scores는 최종 MMR 점수
    """

    # 미리 곡별 user 유사도 계산 (cosine)
    # shape (N,)
    user_sims = np.dot(
        cand_latents,
        user_vec / (np.linalg.norm(user_vec)+1e-12)
    )
    user_sims = user_sims / (
        np.linalg.norm(cand_latents, axis=1)+1e-12
    )

    selected_idx = []
    candidate_idx = list(range(len(cand_ids)))

    for _ in range(min(top_k, len(candidate_idx))):
        best_idx = None
        best_score = -1e9

        for idx_i in candidate_idx:
            sim_to_user = user_sims[idx_i]

            if not selected_idx:
                diversity_penalty = 0.0
            else:
                # 이미 뽑힌 곡들과의 최대 유사도 (겹치면 penalty)
                already_vecs = cand_latents[selected_idx]
                vi = cand_latents[idx_i]

                # cosine 유사도들
                denom_vi = (np.linalg.norm(vi)+1e-12)
                denom_sel = (np.linalg.norm(already_vecs, axis=1)+1e-12)
                sims_to_sel = np.dot(already_vecs, vi / denom_vi) / denom_sel
                max_sim_to_sel = sims_to_sel.max()
                diversity_penalty = max_sim_to_sel

            mmr_score = lambda_div * sim_to_user - (1 - lambda_div) * diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx_i

        selected_idx.append(best_idx)
        candidate_idx.remove(best_idx)

    sel_ids = [cand_ids[i] for i in selected_idx]
    sel_scores = [user_sims[i] for i in selected_idx]  # 보고용으로는 원래 user cosine 점수 줘
    sel_idx = selected_idx

    return sel_ids, sel_scores, sel_idx

# -----------------------
# 추천 생성 & CSV 저장
# -----------------------

def recommend_all_users(
    user_pref,
    user_hist,
    emotion_to_trackinfo,
    df_train_raw,
    emotion_id2text,
):
    """
    유저별 × 감정별 추천 생성.
    - 유저 로그: "추천 중: user {uid}, emotion {eid} (...)"
    - 첫 번째 유저는 top20 프리뷰 출력 (코사인 유사도 포함)
    - 다양성 강화를 위해 MMR 리랭킹 사용
    - 최종 TOP_K_PER_EMOTION = 100 저장

    결과 CSV 컬럼:
      user_id, emotion_id, emotion_text, rank,
      track_uri, score_raw,
      track_name, artists, album_name,
      + FEATURE_COLS
    """
    rows = []
    printed_preview = False

    # train 메타를 곡ID(uri)로 인덱싱해서 조인 쉽게
    df_meta = df_train_raw.set_index(TRACK_ID_TRAIN_COL)

    for uid, emo2vec in user_pref.items():
        for eid, uvec in emo2vec.items():
            if eid not in emotion_to_trackinfo:
                continue

            print(f"[INFO] 추천 중: user {uid}, emotion {eid} ({emotion_id2text.get(eid, f'Emotion {eid}')})")

            track_ids_all = emotion_to_trackinfo[eid]["track_ids"]   # uri list
            latents_all   = emotion_to_trackinfo[eid]["latents"]     # (N_eid, STYLE_DIM)

            # MMR 기반 top-k 선택
            sel_ids, sel_scores, sel_idx = mmr_rerank(
                user_vec=uvec,
                cand_latents=latents_all,
                cand_ids=track_ids_all,
                top_k=TOP_K_PER_EMOTION,
                lambda_div=0.7,  # 유사도 70%, 다양성 30%
            )

            meta_rows = df_meta.loc[sel_ids]

            # 첫 유저 프리뷰 (Top 20)
            if not printed_preview:
                print("\n=== [Preview] First User's Top 20 Recommendations (MMR reranked) ===")
                preview_n = min(20, len(sel_ids))
                for i in range(preview_n):
                    tid = sel_ids[i]
                    sc = sel_scores[i]
                    m = meta_rows.iloc[i]
                    tname = m.get("track_name", "Unknown")
                    arts = m.get("artists", "Unknown")
                    print(f"{i+1:02d}. {tname} - {arts} | cosine={sc:.5f}")
                print("===============================================================\n")
                printed_preview = True

            # CSV rows
            for rank_i, (tid, sc) in enumerate(zip(sel_ids, sel_scores), start=1):
                m = df_meta.loc[tid]

                out_row = {
                    "user_id": uid,
                    "emotion_id": eid,
                    "emotion_text": emotion_id2text.get(eid, f"Emotion {eid}"),
                    "rank": rank_i,
                    "track_uri": tid,
                    "score_raw": sc,
                    "track_name": m.get("track_name", None),
                    "artists": m.get("artists", None),
                    "album_name": m.get("album_name", None),
                }
                for fcol in FEATURE_COLS:
                    out_row[fcol] = m.get(fcol, None)

                rows.append(out_row)

    recs_df = pd.DataFrame(rows)

    # CF로 전달할 CBF 결과 저장
    os.makedirs(CBF_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(CBF_OUTPUT_DIR, f"cbf_recommendations_all_{ts}.csv")

    # cf.cf2.py의 item_id 생성 로직과 동일하게 생성
    # item_id: normalize(track_name) + "_" + normalize(artists)
    recs_df["item_id"] = recs_df["track_name"].str.lower().str.replace(" ", "", regex=False) + \
                         "_" + \
                         recs_df["artists"].str.lower().str.replace(" ", "", regex=False)

    recs_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved CBF recommendations for CF to {output_path}")

# -----------------------
# 감정별 임베딩 시각화 저장
# -----------------------

def visualize_tsne_per_emotion(emotion_to_trackinfo, emotion_id2text):
    """
    감정(eid)별 latent 공간을 따로 t-SNE로 2D로 줄여서
    embedding_tsne_emotion_{eid}.png 로 저장한다.

    각 감정마다 TSNE_PER_EMOTION_SAMPLE만큼 샘플링.
    """

    for eid, pack in emotion_to_trackinfo.items():
        lat = pack["latents"]  # (N_eid, STYLE_DIM)
        n = lat.shape[0]
        if n == 0:
            continue

        take_n = min(n, TSNE_PER_EMOTION_SAMPLE)
        rng = np.random.RandomState(0)
        idx = rng.choice(n, size=take_n, replace=False)

        lat_sample = lat[idx]

        # t-SNE 차원축소
        tsne = TSNE(
            n_components=2,
            init="random",
            random_state=0,
            learning_rate="auto"
        )
        emb2d = tsne.fit_transform(lat_sample)

        plt.figure(figsize=(6, 5))
        plt.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            s=6,
            alpha=0.6,
        )
        plt.title(f"Emotion {eid}: {emotion_id2text.get(eid, f'Emotion {eid}')} latent space")
        plt.tight_layout()

        os.makedirs("output", exist_ok=True)
        out_path = f"output/embedding_tsne_emotion_{eid}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"[DONE] Saved t-SNE plot for emotion {eid} to {out_path}")
