"""
KuaiRand-Pure final submission — LightGBM LambdaRank + past-only historical features.

修复自 agent 的 best solution（val primary 0.6186）：
agent 原版在生成 submission 时用了 sort_values(["date","time_ms"]) 排序后的 test 行序，
导致 row_id 与官方评测集（文件顺序 + date 过滤）不对齐，官方 submit.py --check 会拒绝。
本版保留特征工程不变，只在生成 submission 时用 _orig_idx 恢复「文件顺序」的 test 行序。
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

import sys
sys.path.insert(0, "input")
from evaluate import evaluate

# ----------------------------- Load raw data（保持文件顺序）-----------------------------
train_raw = pd.read_csv("input/log_standard_4_08_to_4_21_pure.csv")
valid_test_raw = pd.read_csv("input/log_standard_4_22_to_5_08_pure.csv")
video_basic = pd.read_csv("input/video_features_basic_pure.csv")

# 关键：记录「文件顺序」的原始行号，供最后恢复官方 test 行序
valid_test_raw["_orig_idx"] = range(len(valid_test_raw))

video_author = video_basic[["video_id", "author_id"]].drop_duplicates("video_id")
train_raw = train_raw.merge(video_author, on="video_id", how="left")
valid_test_raw = valid_test_raw.merge(video_author, on="video_id", how="left")

train_raw["author_id"] = train_raw["author_id"].fillna(-1).astype(int)
valid_test_raw["author_id"] = valid_test_raw["author_id"].fillna(-1).astype(int)

valid_raw = valid_test_raw[valid_test_raw["date"] <= 20220428].copy()
test_raw = valid_test_raw[valid_test_raw["date"] > 20220428].copy()
print(f"Train {train_raw.shape}, Valid {valid_raw.shape}, Test {test_raw.shape}", flush=True)

# ----------------------------- 时间排序算历史特征（cumsum 需要时间顺序）----------------
all_data = pd.concat([train_raw, valid_raw, test_raw], ignore_index=True)
all_data = all_data.sort_values(["date", "time_ms"]).reset_index(drop=True)

for entity in ["user_id", "video_id", "author_id", "tab"]:
    grp = all_data.groupby(entity, sort=False)
    all_data[f"{entity}_total_before"] = grp.cumcount()
    all_data[f"{entity}_lv_before"] = grp["long_view"].cumsum() - all_data["long_view"]
    all_data[f"{entity}_click_before"] = grp["is_click"].cumsum() - all_data["is_click"]

for entity in ["user_id", "video_id", "author_id", "tab"]:
    total = all_data[f"{entity}_total_before"]
    all_data[f"{entity}_lv_rate"] = all_data[f"{entity}_lv_before"] / (total + 1)
    all_data[f"{entity}_click_rate"] = all_data[f"{entity}_click_before"] / (total + 1)

all_data["hour"] = all_data["hourmin"] // 100
all_data["is_weekend"] = (all_data["date"] % 7).isin([5, 6]).astype(int)

train_features = all_data[all_data["date"] <= 20220421].copy()
valid_features = all_data[(all_data["date"] > 20220421) & (all_data["date"] <= 20220428)].copy()

feature_cols = [
    "duration_ms",
    "user_id_total_before", "user_id_lv_rate", "user_id_click_rate",
    "video_id_total_before", "video_id_lv_rate", "video_id_click_rate",
    "author_id_total_before", "author_id_lv_rate",
    "tab_total_before", "tab_lv_rate",
    "hour", "is_weekend",
]
categorical_cols = ["user_id", "video_id", "author_id", "tab"]

for col in categorical_cols:
    for df in (train_features, valid_features):
        df[col] = df[col].astype("category")

# ----------------------------- LightGBM LambdaRank -----------------------------
train_sorted = train_features.sort_values("user_id")
valid_sorted = valid_features.sort_values("user_id")

lgb_train = lgb.Dataset(
    train_sorted[feature_cols + categorical_cols],
    label=train_sorted["long_view"],
    group=train_sorted.groupby("user_id", sort=False).size().values,
    categorical_feature=categorical_cols,
)
lgb_valid = lgb.Dataset(
    valid_sorted[feature_cols + categorical_cols],
    label=valid_sorted["long_view"],
    group=valid_sorted.groupby("user_id", sort=False).size().values,
    reference=lgb_train,
    categorical_feature=categorical_cols,
)

params = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
    "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "min_data_in_leaf": 20, "verbose": -1, "num_threads": 4,
}
model = lgb.train(params, lgb_train, num_boost_round=200,
                  valid_sets=[lgb_valid], callbacks=[lgb.early_stopping(50, verbose=False)])

# ----------------------------- 打分 -----------------------------
valid_scores = model.predict(valid_features[feature_cols + categorical_cols])
r = evaluate(valid_features["user_id"].astype(int).tolist(),
             valid_features["long_view"].tolist(), valid_scores)
print(f"Validation GAUC: {r['GAUC']:.4f}", flush=True)
print(f"Validation nDCG@5: {r['nDCG@5']:.4f}", flush=True)
print(f"Validation primary: {r['primary']:.4f}", flush=True)

# ----------------------------- 生成 submission（恢复官方文件行序）-----------------------------
test_features = all_data[all_data["date"] > 20220428].copy()
for col in categorical_cols:
    test_features[col] = test_features[col].astype("category")
test_scores = model.predict(test_features[feature_cols + categorical_cols])

# 按 _orig_idx（文件顺序）重排，使 row_id 与官方 data.load()['test'] 严格对齐
order = test_features["_orig_idx"].values.argsort()
test_features_final = test_features.iloc[order].reset_index(drop=True)
test_scores_final = test_scores[order]

# 严格对齐官方 submit.py 的 write_submission：
# user_id / video_id 保持字符串，score 用 %.6g（与官方 f"{float(s):.6g}" 一致）
os.makedirs("submission", exist_ok=True)
import csv as _csv
uids = test_features_final["user_id"].astype(str).values
vids = test_features_final["video_id"].astype(str).values
with open("submission/submission.csv", "w", newline="") as fh:
    w = _csv.writer(fh)
    w.writerow(["row_id", "user_id", "video_id", "score"])
    for i in range(len(test_features_final)):
        w.writerow([i, uids[i], vids[i], f"{float(test_scores_final[i]):.6g}"])
print(f"Submission saved: {len(test_features_final)} rows", flush=True)
