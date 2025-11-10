import os
import glob
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix, csr_matrix


# =========================
# 공통 설정
# =========================
EMO_COL = "new_label"  # 훈련셋은 'new_label', 테스트셋은 'predicted_label' 사용

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

# CF 평가 시 최소 곡 수 (user, emotion 그룹별 train/test split 적용 기준)
MIN_CF_ITEMS_PER_GROUP = 5


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def normalize(s) -> str:
    return str(s).lower().replace(" ", "")





# =========================
# 0. 데이터 로드 & 전처리 (머지, item_id 생성)
# =========================
def load_and_prepare_user_dataset(
    user_log_path: Optional[str] = None,
    label_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    - 유저 로그와 라벨/피처 데이터(train_labeled_dataset)를 merge_key로 병합
    - item_id 생성: normalize(track_name) + "_" + normalize(artists)
    - 주의: user_id는 원본 값을 유지(이미 숫자라면 그대로 사용).
    """
    if user_log_path is None:
        user_log_path = os.path.join("data", "user_dataset_filtered.csv")
    if label_path is None:
        label_path = os.path.join("data", "train_labeled_dataset.csv")

    print("=== [0] 데이터 로드 & 전처리 (머지) ===")
    df = pd.read_csv(user_log_path)
    label_df = pd.read_csv(label_path)

    print(f"원본 유저 로그 크기: {df.shape}")
    print(f"라벨/피처 데이터 크기: {label_df.shape}")

    # 병합 키 생성
    df["merge_key"] = df["track_name"].apply(normalize) + df["artists"].apply(normalize)
    label_df["merge_key"] = label_df["track_name"].apply(normalize) + label_df["artists"].apply(normalize)

    # 라벨 데이터는 곡 중복 제거 후 머지
    label_df_dedup = label_df.drop_duplicates(subset=["merge_key"], keep="first")
    cols_to_add = FEATURE_COLS + [EMO_COL]

    merged = df.merge(
        label_df_dedup[["merge_key"] + cols_to_add],
        on="merge_key",
        how="inner",
    ).drop(columns=["merge_key"]).copy()

    # item_id 생성
    merged["item_id"] = (
        merged["track_name"].apply(normalize) + "_" + merged["artists"].apply(normalize)
    )

    # user_id는 가급적 원본 숫자 형태 유지. 숫자가 아니면 카테고리 인코딩.
    # user_id dtype 확인 (pandas ExtensionDtype 대응)
    if not pd.api.types.is_integer_dtype(merged["user_id"].dtype):
        merged["user_id_raw"] = merged["user_id"].astype(str)
        cat = merged["user_id_raw"].astype("category")
        merged["user_id"] = cat.cat_codes.astype(int)
        print("user_id가 비수치형 → category 인코딩 적용")

    print(f"최종 병합 데이터 shape: {merged.shape}")
    print(f"고유 유저 수: {merged['user_id'].nunique()}, 고유 아이템 수: {merged['item_id'].nunique()}")
    return merged


# =========================
# 1. 최신 산출물(CBF/CF 데이터셋) 경로 탐색
# =========================
def find_latest_file(dir_path: str, pattern: str) -> Optional[str]:
    paths = glob.glob(os.path.join(dir_path, pattern))
    if not paths:
        return None
    paths.sort()
    return paths[-1]


def load_cbf_recommendations(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    if path is None:
        path = find_latest_file("output/cbf_intermediate", "cbf_recommendations_all_*.csv")
    if path and os.path.exists(path):
        print(f"CBF 추천 파일 로드: {path}")
        return pd.read_csv(path)
    print("CBF 추천 파일을 찾지 못했습니다.")
    return None


# =========================
# 2. 유저-아이템 행렬 생성 (감정 필터 + 선택 아이템 필터)
# =========================
def build_user_item_matrix(
    df: pd.DataFrame,
    emotion: str,
    allowed_items: Optional[set] = None,
) -> Tuple[Any, List[int], List[str], Dict[int, int], Dict[str, int]]:
    """
    df: 병합 유저 로그(+라벨) 데이터
    emotion: 선택 감정만 사용
    allowed_items: None이 아니면 해당 item_id 집합으로 컬럼을 제한

    return: (X, user_ids, item_ids, uid2row, iid2col)
    - X: shape (n_users, n_items) CSR, 값은 binary implicit(시청/청취)
    """
    use_df = df[df[EMO_COL] == emotion].copy()
    if allowed_items is not None:
        use_df = use_df[use_df["item_id"].isin(allowed_items)]

    if use_df.empty:
        return csr_matrix((0, 0), dtype=np.float32), [], [], {}, {}

    user_ids = sorted(use_df["user_id"].unique().tolist())
    item_ids = sorted(use_df["item_id"].unique().tolist())
    uid2row = {u: i for i, u in enumerate(user_ids)}
    iid2col = {it: j for j, it in enumerate(item_ids)}

    rows = use_df["user_id"].map(uid2row).values
    cols = use_df["item_id"].map(iid2col).values
    data = np.ones(len(use_df), dtype=np.float32)

    # SciPy 버전에 따라 .tocsr()가 csr_matrix 대신 새 csr_array 타입을 반환할 수 있어 변수 타입 주석을 제거
    X = coo_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids))).tocsr()
    return X, user_ids, item_ids, uid2row, iid2col


# =========================
# 2-1. CF용 Train/Test Split (user, emotion 단위)
# =========================
def split_cf_train_test_per_user_emotion(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    min_items: int = MIN_CF_ITEMS_PER_GROUP,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    (user_id, emotion) 그룹별로 아이템(행) 분할.
    - 그룹 내 아이템 수가 min_items 미만이면 해당 그룹은 통째로 train으로 편입(평가 제외)
    - 충분하면 train/test로 분리.
    반환: train_df, test_df
    """
    train_list, test_list = [], []
    total = eval_groups = 0
    for (uid, emo), g in df.groupby(["user_id", EMO_COL]):
        total += 1
        if len(g) < min_items:
            train_list.append(g)
            continue
        # 재현성 있는 분할
        g_shuffled = g.sample(frac=1.0, random_state=seed)
        split_idx = int(len(g_shuffled) * (1 - test_size))
        tr_g = g_shuffled.iloc[:split_idx]
        te_g = g_shuffled.iloc[split_idx:]
        # test가 1개 미만이면 평가 불안정 → 통째로 train
        if te_g.empty:
            train_list.append(g)
            continue
        train_list.append(tr_g)
        test_list.append(te_g)
        eval_groups += 1
    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=df.columns)
    print(f"CF Train/Test Split 완료: 총 그룹 {total}, 평가 대상 그룹 {eval_groups}, train 행 {len(train_df)}, test 행 {len(test_df)}")
    return train_df, test_df


# =========================
# 2-2. 평가 지표 함수 (CBF와 동일 형식)
# =========================
def _dcg_at_k(rel_list: List[int], k: int) -> float:
    rel = np.asarray(rel_list)[:k]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel / discounts))

def _ndcg_at_k(rel_list: List[int], k: int) -> float:
    dcg = _dcg_at_k(rel_list, k)
    ideal = sorted(rel_list, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


# =========================
# 2-3. CF 평가 로직
# =========================
def evaluate_cf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_k: int,
    neighbor_k: int = 50,
) -> Tuple[Dict[str, float], List[dict]]:
    """
    CF 추천 평가:
    - train_df로 사용자-아이템 행렬 구성 (emotion별 캐시)
    - 각 test (user, emotion) 그룹에 대해 추천 생성 (train 상호작용 제외)
    - test 그룹의 item_id 집합을 GT로 사용, 추천과 비교하여 HitRate/Precision/Recall/NDCG 측정
    - 실패 케이스는 fail_cases에 stage='eval_cf'로 기록
    """
    if test_df.empty:
        print("평가 대상 test 데이터가 없습니다. 평가를 건너뜁니다.")
        return {"num_eval_cases": 0, f"HitRate@{top_k}": 0.0, f"Precision@{top_k}": 0.0, f"Recall@{top_k}": 0.0, f"NDCG@{top_k}": 0.0}, []

    # 캐시 준비
    matrix_cache: Dict[str, Tuple[Any, List[int], List[str], Dict[int, int], Dict[str, int]]] = {}
    fail_cases: List[dict] = []
    hit_list: List[float] = []
    prec_list: List[float] = []
    rec_list: List[float] = []
    ndcg_list: List[float] = []

    # 사용자 감정별 train 소비 아이템 set (추천에서 제외)
    consumed_train: Dict[Tuple[int, str], set] = {
        (uid, emo): set(g["item_id"].astype(str))
        for (uid, emo), g in train_df.groupby(["user_id", EMO_COL])
    }

    eval_groups = list(test_df.groupby(["user_id", EMO_COL]))
    for idx, ((uid, emo), g_test) in enumerate(eval_groups, start=1):
        gt_items = set(g_test["item_id"].astype(str))
        if not gt_items:
            continue

        # 행렬 캐시 생성 (emotion 전체 train 기반)
        if emo not in matrix_cache:
            matrix_cache[emo] = build_user_item_matrix(train_df, emotion=emo, allowed_items=None)
        X, user_ids, item_ids, uid2row, iid2col = matrix_cache[emo]

        if uid not in uid2row or X.shape[1] == 0:
            fail_cases.append({"user_id": uid, EMO_COL: emo, "stage": "eval_cf", "reason": "user_or_empty_matrix"})
            continue

        # 추천 생성 (train 상호작용만으로 학습된 행렬 사용)
        rec_pairs = recommend_cf_for_user(
            user_id=uid,
            emotion=emo,
            X=X,
            user_ids=user_ids,
            item_ids=item_ids,
            uid2row=uid2row,
            iid2col=iid2col,
            top_k=top_k,
            neighbor_k=neighbor_k,
        )

        if not rec_pairs:
            fail_cases.append({"user_id": uid, EMO_COL: emo, "stage": "eval_cf", "reason": "no_recs"})
            continue

        rec_item_ids = [iid for iid, _ in rec_pairs]
        rel_list = [1 if iid in gt_items else 0 for iid in rec_item_ids]
        hit_list.append(1.0 if any(rel_list) else 0.0)
        prec_list.append(sum(rel_list) / len(rec_item_ids))
        rec_list.append(sum(rel_list) / len(gt_items))
        ndcg_list.append(_ndcg_at_k(rel_list, top_k))

        if idx % 1000 == 0:
            print(f"  CF 평가 진행: {idx}/{len(eval_groups)}")

    metrics = {
        "num_eval_cases": len(hit_list),
        f"HitRate@{top_k}": float(np.mean(hit_list)) if hit_list else 0.0,
        f"Precision@{top_k}": float(np.mean(prec_list)) if prec_list else 0.0,
        f"Recall@{top_k}": float(np.mean(rec_list)) if rec_list else 0.0,
        f"NDCG@{top_k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
    }
    print("\n=== [CF 평가 결과] ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    return metrics, fail_cases


# =========================
# 3. 사용자 기반 CF 추천
# =========================
def recommend_cf_for_user(
    user_id: int,
    emotion: str,
    X: Any,
    user_ids: List[int],
    item_ids: List[str],
    uid2row: Dict[int, int],
    iid2col: Dict[str, int],
    top_k: int = 30,
    neighbor_k: int = 50,
) -> List[Tuple[str, float]]:
    if user_id not in uid2row:
        return []

    if X.shape[1] == 0:
        return []

    uidx = uid2row[user_id]

    # 유사도 계산 (희소행렬 기반)
    uvec = X.getrow(uidx)
    if uvec.nnz == 0:
        return []

    # 활성 사용자(현재 아이템 서브행렬에서 한 개 이상 상호작용이 있는 사용자)만 대상으로 유사도 계산
    # 활성 사용자가 아니면 코사인 유사도는 0이므로 계산에서 제외해도 동일 결과
    try:
        row_nnz = np.asarray(X.getnnz(axis=1)).ravel()
    except Exception:
        # 일부 SciPy 버전 호환: axis=1 반환 형식이 다를 수 있어 보정
        row_nnz = X.getnnz(axis=1)
        if not isinstance(row_nnz, np.ndarray):
            row_nnz = np.array(row_nnz).ravel()

    active_idx = np.where(row_nnz > 0)[0]
    if active_idx.size == 0:
        return []

    subX = X[active_idx, :]
    sims_sub = cosine_similarity(uvec, subX).flatten()
    sims = np.zeros(X.shape[0], dtype=np.float32)
    sims[active_idx] = sims_sub
    sims[uidx] = 0.0  # 자기 자신 제외

    # 상위 neighbor_k 이웃만 사용하여 가중치 합
    if neighbor_k is not None and neighbor_k < len(sims):
        top_n_idx = np.argpartition(-sims, neighbor_k)[:neighbor_k]
        sims_mask = np.zeros_like(sims)
        sims_mask[top_n_idx] = sims[top_n_idx]
        sims = sims_mask

    # 아이템 스코어 = X.T @ sims
    scores = X.T.dot(sims)  # shape: (n_items,)

    # 이미 소비한 아이템 제외
    consumed_cols = set(uvec.indices.tolist())
    all_cols = np.arange(X.shape[1])
    cand_cols = [c for c in all_cols if c not in consumed_cols]
    if not cand_cols:
        return []

    cand_scores = scores[cand_cols]
    if len(cand_cols) <= top_k:
        top_idx_local = np.argsort(-cand_scores)
    else:
        part = np.argpartition(-cand_scores, top_k)[:top_k]
        top_idx_local = part[np.argsort(-cand_scores[part])]

    result: List[Tuple[str, float]] = []
    for pos in top_idx_local:
        col = cand_cols[pos]
        item_id = item_ids[col]
        score = float(cand_scores[pos])
        result.append((item_id, score))
    return result


# =========================
# 4. 메인 로직: 케이스 분기 및 저장
# =========================
def run_cf(
    user_id: int,
    emotion: str,
    top_k: int = 30,
    user_log_path: Optional[str] = None,
    label_path: Optional[str] = None,
    cbf_recs_path: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """
    - CBF 추천 파일에 (user, emotion) 존재 → 그 100곡만 후보로 CF 수행
    - 아니면 전체(해당 emotion) 매트릭스로 CF 수행 (fallback)
    - 결과 CSV 경로 반환
    """
    merged = load_and_prepare_user_dataset(user_log_path, label_path)

    # 메타: item_id → (track_name, artists)
    item_meta = (
        merged.drop_duplicates("item_id")[["item_id", "track_name", "artists", EMO_COL]]
        .set_index("item_id")
        .to_dict(orient="index")
    )

    # 산출물 로드
    cbf_recs = load_cbf_recommendations(cbf_recs_path)

    # 케이스 A: CBF 추천 내 존재하면 해당 100곡만 후보
    allowed_items: Optional[set] = None
    source = "CF_All"

    if cbf_recs is not None:
        # cbf 파일은 최소한 'user_id','emotion' 또는 EMO_COL, 'item_id' 컬럼을 갖는다고 가정
        emo_col_in_cbf = EMO_COL if EMO_COL in cbf_recs.columns else "emotion"
        cbf_slice = cbf_recs[(cbf_recs["user_id"] == user_id) & (cbf_recs[emo_col_in_cbf] == emotion)]
        if not cbf_slice.empty:
            allowed_items = set(cbf_slice["item_id"].astype(str).tolist())
            source = "CBF100"
            print(f"CBF 추천 상위 100곡 사용 (user={user_id}, emo={emotion}, 후보수={len(allowed_items)})")

    # 케이스 B: CBF 추천 없으면 전체(해당 emotion) 매트릭스 사용
    if allowed_items is None:
        print(f"CBF 추천 없음 (user={user_id}, emo={emotion}), 전체 매트릭스 사용")
        source = "CF_All"

    # 행렬 구성
    # - CBF 추천이 존재하면: 해당 100곡만 포함하되,
    #   만약 사용자-감정 교집합이 전무하면 사용자-감정 소비 곡을 추가 포함하여 서브행렬 구성
    # - CBF 추천이 없으면: 전체(해당 emotion) 매트릭스 사용
    allowed_for_matrix = allowed_items
    if allowed_items is not None:
        user_emo_items = set(
            merged[(merged["user_id"] == user_id) & (merged[EMO_COL] == emotion)]["item_id"].astype(str).tolist()
        )
        if allowed_items.isdisjoint(user_emo_items):
            # 100곡과 사용자-감정 소비곡 간 교집합이 없으면, 사용자 곡을 추가하여 cold-start 방지
            allowed_for_matrix = set(allowed_items) | user_emo_items

    X, user_ids, item_ids, uid2row, iid2col = build_user_item_matrix(
        merged, emotion=emotion, allowed_items=allowed_for_matrix
    )

    if X.shape[0] == 0 or X.shape[1] == 0:
        print("해당 조건으로 구성된 유저-아이템 매트릭스가 비어 있습니다. 추천 불가.")
        # 빈 결과라도 형식에 맞게 저장
        out_dir = out_dir or "output/final_recommendations"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"cf_rec_user_{user_id}_{emotion}_{now_ts()}.csv")
        pd.DataFrame([], columns=[
            "user_id", EMO_COL, "rank", "item_id", "track_name", "artists", "score", "source"
        ]).to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    # 후보가 CBF 100곡이면 충분한 수집을 위해 넉넉히 받아온 다음 사후 필터링
    req_top_k = top_k
    cand_top_k = max(req_top_k * 5, 500) if allowed_items is not None else req_top_k


    rec_pairs = recommend_cf_for_user(
        user_id=user_id,
        emotion=emotion,
        X=X,
        user_ids=user_ids,
        item_ids=item_ids,
        uid2row=uid2row,
        iid2col=iid2col,
        top_k=cand_top_k,
        neighbor_k=50,
    )

    # 사후 필터링: 이미 서브행렬에서 계산했지만 혹시 모를 누락 방지
    if allowed_items is not None:
        rec_pairs = [p for p in rec_pairs if p[0] in allowed_items][:req_top_k]

    # CF 결과가 비었으면 간단 인기 기준으로 대체 (동일 감정에서의 전역 인기, 이미 소비한 곡 제외)
    if not rec_pairs:
        print("CF 결과가 비었습니다. 전역 인기(감정별) 기반으로 대체합니다.")
        emo_df = merged[merged[EMO_COL] == emotion]
        # 사용자가 이미 소비한 아이템
        consumed_user = set(emo_df[emo_df["user_id"] == user_id]["item_id"].astype(str).tolist())
        pop_counts = (
            emo_df.groupby("item_id").size().sort_values(ascending=False)
        )
        pop_items = pop_counts.index.astype(str).tolist()
        # CBF 제한이 있으면 교집합만, 사용자가 들은 곡 제외
        filtered = [iid for iid in pop_items if (allowed_items is None or iid in allowed_items) and iid not in consumed_user]
        rec_pairs = [(iid, float(pop_counts.loc[iid])) for iid in filtered[:req_top_k]]

    # 결과 구성 및 저장
    rows = []
    for rank, (iid, score) in enumerate(rec_pairs[:req_top_k], start=1):
        meta = item_meta.get(iid, {})
        rows.append({
            "user_id": user_id,
            EMO_COL: emotion,
            "rank": rank,
            "item_id": iid,
            "track_name": meta.get("track_name"),
            "artists": meta.get("artists"),
            "score": score,
            "source": source,
        })

    out_dir = out_dir or "output/final_recommendations"
    os.makedirs(out_dir, exist_ok=True)
    ts = now_ts()
    out_path = os.path.join(out_dir, f"cf_rec_user_{user_id}_{emotion}_{ts}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"추천 결과 저장: {out_path} (총 {len(rows)}건)")
    return out_path


# =========================
# 4-1. 배치 모드: 전체 유저×감정 조합 추천 생성 및 단일 파일 저장
# =========================
def run_hybrid_cf(
    top_k: int = 30,
    user_log_path: Optional[str] = None,
    label_path: Optional[str] = None,
    cbf_recs_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    eval_mode: bool = False,
    test_size: float = 0.2,
    min_cf_items: int = MIN_CF_ITEMS_PER_GROUP,
    seed: int = 42,
) -> str:
    print("=== [Batch] 데이터 로드 ===")
    merged = load_and_prepare_user_dataset(user_log_path, label_path)

    # 평가 모드이면 train/test split 수행
    if eval_mode:
        train_df, test_df = split_cf_train_test_per_user_emotion(
            merged, test_size=test_size, seed=seed, min_items=min_cf_items
        )
        cf_eval_metrics, cf_fail_cases = evaluate_cf(train_df, test_df, top_k=top_k, neighbor_k=50)
        ts_eval = now_ts()
        out_dir_use = out_dir or "output/final_recommendations"
        os.makedirs(out_dir_use, exist_ok=True)
        eval_path = os.path.join(out_dir_use, f"cf_eval_result_{ts_eval}.csv")
        fail_path = os.path.join(out_dir_use, f"cf_fail_cases_{ts_eval}.csv")
        pd.DataFrame([cf_eval_metrics]).to_csv(eval_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(cf_fail_cases).to_csv(fail_path, index=False, encoding="utf-8-sig")
        print(f"CF 평가 결과 저장: {eval_path}")
        print(f"CF 실패 케이스 저장: {fail_path}")
        # 추천은 기존 동작과 동일하게 전체 merged 데이터 기반으로 진행
        base_df_for_recs = merged
    else:
        base_df_for_recs = merged

    # 메타 정보 구축
    item_meta = (
        merged.drop_duplicates("item_id")[["item_id", "track_name", "artists", EMO_COL]]
        .set_index("item_id")
        .to_dict(orient="index")
    )

    # 산출물 로드
    cbf_recs = load_cbf_recommendations(cbf_recs_path)

    # 전체 (user, emotion) 조합
    combos = base_df_for_recs[["user_id", EMO_COL]].drop_duplicates().values.tolist()
    print(f"총 (user, emotion) 조합 수: {len(combos)}")

    # 배치 성능 최적화 캐시
    # - 감정별 전역 인기(popularity) 캐시
    pop_counts_cache: Dict[str, pd.Series] = {}
    # - 감정별 전체 행렬(allowed_items=None) 캐시: 반복 생성 방지
    matrix_cache: Dict[str, Tuple[Any, List[int], List[str], Dict[int, int], Dict[str, int]]] = {}
    # - 사용자×감정 청취 곡 set 캐시
    listened_dict: Dict[Tuple[int, str], set] = {
        (uid, emo): set(g["item_id"].astype(str))
        for (uid, emo), g in base_df_for_recs.groupby(["user_id", EMO_COL])
    }

    all_rows: List[dict] = []
    processed = 0

    for user_id, emotion in combos:
        processed += 1

        # allowed_items/source 결정
        allowed_items: Optional[set] = None
        source = "CF_All"
        if cbf_recs is not None:
            emo_col_in_cbf = EMO_COL if EMO_COL in cbf_recs.columns else "emotion"
            cbf_slice = cbf_recs[(cbf_recs["user_id"] == user_id) & (cbf_recs[emo_col_in_cbf] == emotion)]
            if not cbf_slice.empty:
                allowed_items = set(cbf_slice["item_id"].astype(str).tolist())
                source = "CBF100"

        if allowed_items is None:
            source = "CF_All"

        # CBF100과 사용자 감정 소비곡 교집합 점검 → 없으면 사용자 곡 추가하여 서브행렬 구성
        allowed_for_matrix = allowed_items
        if allowed_items is not None:
            user_emo_items = listened_dict.get((user_id, emotion), set())
            if allowed_items.isdisjoint(user_emo_items):
                allowed_for_matrix = set(allowed_items) | user_emo_items

        # 행렬 구성 (캐시 + 컬럼 슬라이싱 최적화)
        if emotion not in matrix_cache:
            matrix_cache[emotion] = build_user_item_matrix(
                base_df_for_recs, emotion=emotion, allowed_items=None
            )
        fullX, full_user_ids, full_item_ids, full_uid2row, full_iid2col = matrix_cache[emotion]

        if allowed_for_matrix is None:
            # 전체 감정 행렬 재사용
            X = fullX
            user_ids = full_user_ids
            item_ids = full_item_ids
            uid2row = full_uid2row
            iid2col = full_iid2col
        else:
            # 허용 아이템 서브셋에 대해 컬럼 슬라이스 (재구성 없이 성능 향상)
            sub_col_indices = [full_iid2col[iid] for iid in allowed_for_matrix if iid in full_iid2col]
            if not sub_col_indices:
                # 허용된 아이템이 전혀 매트릭스에 없으면 빈 행렬 처리
                X = csr_matrix((fullX.shape[0], 0), dtype=np.float32)
                user_ids = full_user_ids
                item_ids = []
                uid2row = full_uid2row
                iid2col = {}
            else:
                # 컬럼 슬라이싱 (CSR → 효율적)
                X = fullX[:, sub_col_indices]
                user_ids = full_user_ids
                item_ids = [full_item_ids[c] for c in sub_col_indices]
                uid2row = full_uid2row
                iid2col = {iid: idx for idx, iid in enumerate(item_ids)}

        req_top_k = top_k
        cand_top_k = req_top_k
        rows: List[dict] = []

        # 안전 접근: 우선 None/유저불포함 검사 후, 별도로 shape 검사
        need_pop_fallback = False
        if (X is None) or (user_id not in uid2row):
            need_pop_fallback = True
        else:
            rows_num, cols_num = getattr(X, 'shape', (0, 0))
            if (rows_num == 0) or (cols_num == 0):
                need_pop_fallback = True

        if need_pop_fallback:
            emo_df = base_df_for_recs[base_df_for_recs[EMO_COL] == emotion]
            consumed_user = listened_dict.get((user_id, emotion), set())
            if emotion not in pop_counts_cache:
                pop_counts_cache[emotion] = emo_df.groupby("item_id").size().sort_values(ascending=False)
            pop_counts = pop_counts_cache[emotion]
            pop_items = pop_counts.index.astype(str).tolist()
            filtered = [iid for iid in pop_items if (allowed_items is None or iid in allowed_items) and iid not in consumed_user]
            chosen = filtered[:req_top_k]
            for rank, iid in enumerate(chosen, start=1):
                meta = item_meta.get(iid, {})
                rows.append({
                    "user_id": user_id,
                    EMO_COL: emotion,
                    "rank": rank,
                    "item_id": iid,
                    "track_name": meta.get("track_name"),
                    "artists": meta.get("artists"),
                    "score": float(pop_counts.loc[iid]) if iid in pop_counts.index else 0.0,
                    "source": source if allowed_items is None else "CBF100_POP_FALLBACK",
                })
        else:
            rec_pairs = recommend_cf_for_user(
                user_id=user_id,
                emotion=emotion,
                X=X,
                user_ids=user_ids,
                item_ids=item_ids,
                uid2row=uid2row,
                iid2col=iid2col,
                top_k=cand_top_k,
                neighbor_k=50,
            )
            if allowed_items is not None:
                rec_pairs = [p for p in rec_pairs if p[0] in allowed_items][:req_top_k]
            else:
                rec_pairs = rec_pairs[:req_top_k]

            if not rec_pairs:
                emo_df = base_df_for_recs[base_df_for_recs[EMO_COL] == emotion]
                consumed_user = listened_dict.get((user_id, emotion), set())
                if emotion not in pop_counts_cache:
                    pop_counts_cache[emotion] = emo_df.groupby("item_id").size().sort_values(ascending=False)
                pop_counts = pop_counts_cache[emotion]
                pop_items = pop_counts.index.astype(str).tolist()
                filtered = [iid for iid in pop_items if (allowed_items is None or iid in allowed_items) and iid not in consumed_user]
                chosen = filtered[:req_top_k]
                for rank, iid in enumerate(chosen, start=1):
                    meta = item_meta.get(iid, {})
                    rows.append({
                        "user_id": user_id,
                        EMO_COL: emotion,
                        "rank": rank,
                        "item_id": iid,
                        "track_name": meta.get("track_name"),
                        "artists": meta.get("artists"),
                        "score": float(pop_counts.loc[iid]) if iid in pop_counts.index else 0.0,
                        "source": source if allowed_items is None else "CBF100_POP_FALLBACK",
                    })
            else:
                for rank, (iid, score) in enumerate(rec_pairs, start=1):
                    meta = item_meta.get(iid, {})
                    rows.append({
                        "user_id": user_id,
                        EMO_COL: emotion,
                        "rank": rank,
                        "item_id": iid,
                        "track_name": meta.get("track_name"),
                        "artists": meta.get("artists"),
                        "score": score,
                        "source": source,
                    })

        all_rows.extend(rows)

        if processed % 1000 == 0:
            print(f"  진행: {processed}/{len(combos)} 조합 처리 완료 (누적 행 {len(all_rows)})")

    out_dir = out_dir or "output/final_recommendations"
    os.makedirs(out_dir, exist_ok=True)
    ts = now_ts()
    out_path = os.path.join(out_dir, f"cf_batch_recommendations_{ts}.csv")
    df_out = pd.DataFrame(all_rows)
    if "user_id" in df_out.columns:
        # user_id 기준 안정 정렬로 저장 (intra-user 기존 순서 유지)
        df_out = df_out.sort_values(by=["user_id"], kind="mergesort", ascending=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"배치 추천 결과 저장: {out_path} (총 {len(df_out)}행, 조합 {len(combos)}개, user_id 기준 정렬)")
    if eval_mode:
        print("평가 모드: cf_eval_result / cf_fail_cases 파일도 함께 생성되었습니다.")
    return out_path

# =========================
# 5. CLI 엔트리포인트
# =========================
def _build_argparser():
    p = argparse.ArgumentParser(description="Hybrid CF stage: CBF 100 제한 또는 CF full로 추천")
    p.add_argument("--user_id", type=int, help="대상 유저 ID (정수). 배치 모드에서는 사용 안 함.")
    p.add_argument("--emotion", type=str, help="감정 라벨. 배치 모드에서는 사용 안 함.")
    p.add_argument("--top_k", type=int, default=30, help="추천 개수")
    p.add_argument("--user_log_path", type=str, default="data/user_dataset_filtered.csv", help="유저 데이터셋")
    p.add_argument("--label_path", type=str, default="data/train_labeled_dataset.csv", help="곡 데이터셋")
    p.add_argument("--cbf_recs_path", type=str, default=None, help="CBF 추천 파일 경로 (기본: 자동 탐색)")
    p.add_argument("--out_dir", type=str, default="output/final_recommendations", help="결과 저장 디렉터리")
    p.add_argument("--batch", action="store_true", help="전체 (user, emotion) 조합 배치 추천 수행 (main.py에서 사용)")
    p.add_argument("--eval", action="store_true", help="배치 모드에서 CF 추천 평가(Train/Test split) 수행 후 결과 저장")
    p.add_argument("--test_size", type=float, default=0.2, help="CF 평가용 test 비율 (기본 0.2)")
    p.add_argument("--min_cf_items", type=int, default=MIN_CF_ITEMS_PER_GROUP, help="CF 평가 대상 최소 곡 수 (기본 5)")
    p.add_argument("--seed", type=int, default=42, help="Train/Test split 시드")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.batch:
        run_hybrid_cf(
            top_k=args.top_k,
            user_log_path=args.user_log_path,
            label_path=args.label_path,
            cbf_recs_path=args.cbf_recs_path,
            out_dir=args.out_dir,
            eval_mode=args.eval,
            test_size=args.test_size,
            min_cf_items=args.min_cf_items,
            seed=args.seed,
        )
    else:
        if args.user_id is None or args.emotion is None:
            raise ValueError("배치 모드가 아닐 경우 --user_id와 --emotion은 필수입니다.")
        run_cf(
            user_id=args.user_id,
            emotion=args.emotion,
            top_k=args.top_k,
            user_log_path=args.user_log_path,
            label_path=args.label_path,
            cbf_recs_path=args.cbf_recs_path,
            out_dir=args.out_dir,
        )
