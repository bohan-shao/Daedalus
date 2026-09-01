import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "input")
from evaluate import evaluate

DATA_DIR = "input"
TRAIN_CSV = os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_1k.csv")
VALID_TEST_CSV = os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_1k.csv")
VIDEO_FEAT_CSV = os.path.join(DATA_DIR, "video_features_basic_1k.csv")

VALID_START = 20220422
VALID_END = 20220428
TEST_START = 20220429
TEST_END = 20220508

print("Loading data...", flush=True)
t0 = time.time()

cols = [
    "date",
    "user_id",
    "video_id",
    "tab",
    "duration_ms",
    "time_ms",
    "long_view",
    "is_click",
]
train_df = pd.read_csv(TRAIN_CSV, usecols=cols)
vt_df = pd.read_csv(VALID_TEST_CSV, usecols=cols)

video_feat = pd.read_csv(VIDEO_FEAT_CSV, usecols=["video_id", "author_id"])
video_feat = video_feat.drop_duplicates(subset=["video_id"], keep="first")

train_df = train_df.merge(video_feat, on="video_id", how="left")
vt_df = vt_df.merge(video_feat, on="video_id", how="left")

train_df["author_id"] = train_df["author_id"].fillna(-1).astype(np.int64)
vt_df["author_id"] = vt_df["author_id"].fillna(-1).astype(np.int64)

print(
    f"Loaded train {train_df.shape}, valid+test {vt_df.shape} in {time.time()-t0:.1f}s",
    flush=True,
)

valid_df = vt_df[(vt_df["date"] >= VALID_START) & (vt_df["date"] <= VALID_END)].copy()
test_df = vt_df[(vt_df["date"] >= TEST_START) & (vt_df["date"] <= TEST_END)].copy()
# Keep the original file-order row index so submission row_id matches data.load()["test"]
test_df["_orig_idx"] = test_df.index.values
del vt_df, video_feat

print(f"Valid rows: {len(valid_df)}, Test rows: {len(test_df)}", flush=True)

train_df["split"] = "train"
valid_df["split"] = "valid"
test_df["split"] = "test"

all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
all_df = all_df.sort_values(["time_ms", "user_id"]).reset_index(drop=True)

print("Computing cumulative historical features...", flush=True)

HALF_LIFE_MS = 2 * 24 * 3600 * 1000
DECAY = 0.5 ** (1.0 / HALF_LIFE_MS)

user_stats = {}
video_stats = {}
author_stats = {}
tab_stats = {}
user_video_stats = {}
user_author_stats = {}
user_tab_stats = {}

# New: separate stats for clicked vs non-clicked impressions
user_click_stats = {}
video_click_stats = {}
author_click_stats = {}
user_video_click_stats = {}

n = len(all_df)
user_hist_rate = np.zeros(n, dtype=np.float32)
user_hist_count = np.zeros(n, dtype=np.float32)
video_hist_rate = np.zeros(n, dtype=np.float32)
video_hist_count = np.zeros(n, dtype=np.float32)
author_hist_rate = np.zeros(n, dtype=np.float32)
author_hist_count = np.zeros(n, dtype=np.float32)
tab_hist_rate = np.zeros(n, dtype=np.float32)
tab_hist_count = np.zeros(n, dtype=np.float32)
user_video_hist_rate = np.zeros(n, dtype=np.float32)
user_video_hist_count = np.zeros(n, dtype=np.float32)
user_author_hist_rate = np.zeros(n, dtype=np.float32)
user_author_hist_count = np.zeros(n, dtype=np.float32)
user_tab_hist_rate = np.zeros(n, dtype=np.float32)
user_tab_hist_count = np.zeros(n, dtype=np.float32)

# New features
user_click_rate = np.zeros(n, dtype=np.float32)
video_click_rate = np.zeros(n, dtype=np.float32)
author_click_rate = np.zeros(n, dtype=np.float32)
user_video_click_rate = np.zeros(n, dtype=np.float32)
user_click_count = np.zeros(n, dtype=np.float32)
video_click_count = np.zeros(n, dtype=np.float32)
author_click_count = np.zeros(n, dtype=np.float32)
user_video_click_count = np.zeros(n, dtype=np.float32)

# Recent window features (last 20 interactions per user)
recent_pos_rate = np.zeros(n, dtype=np.float32)
recent_click_rate = np.zeros(n, dtype=np.float32)
recent_activity_gap = np.zeros(n, dtype=np.float32)

video_duration_bucket = np.zeros(n, dtype=np.int32)
author_popularity = np.zeros(n, dtype=np.float32)
video_age_ms = np.zeros(n, dtype=np.float32)
user_activity_count = np.zeros(n, dtype=np.float32)
video_exposure_total = np.zeros(n, dtype=np.float32)

user_ids = all_df["user_id"].values
video_ids = all_df["video_id"].values
author_ids = all_df["author_id"].values
tab_ids = all_df["tab"].values
long_views = all_df["long_view"].values
is_clicks = all_df["is_click"].values
time_ms = all_df["time_ms"].values
duration_ms = all_df["duration_ms"].values

duration_bins = [
    0,
    5000,
    10000,
    20000,
    30000,
    60000,
    120000,
    300000,
    600000,
    1200000,
    3600000,
    np.inf,
]
video_duration_bucket = np.digitize(duration_ms, duration_bins) - 1

video_first_seen = {}
author_exposure = {}

# Per-user recent history buffers
user_recent_buffer = {}  # user_id -> list of (time_ms, long_view, is_click)


def get_decayed_stats(stats_dict, key, current_time):
    if key not in stats_dict:
        return 0.0, 0.0
    weighted_sum, weight, count, last_time = stats_dict[key]
    if weight <= 0:
        return 0.0, 0.0
    dt = current_time - last_time
    if dt > 0:
        decay = DECAY ** (dt / HALF_LIFE_MS)
        weighted_sum *= decay
        weight *= decay
    rate = weighted_sum / weight if weight > 0 else 0.0
    return rate, count


def update_stats(stats_dict, key, current_time, label):
    if key not in stats_dict:
        stats_dict[key] = [0.0, 0.0, 0, current_time]
    weighted_sum, weight, count, last_time = stats_dict[key]
    dt = current_time - last_time
    if dt > 0 and weight > 0:
        decay = DECAY ** (dt / HALF_LIFE_MS)
        weighted_sum *= decay
        weight *= decay
    weighted_sum += label
    weight += 1.0
    count += 1
    stats_dict[key] = [weighted_sum, weight, count, current_time]


for i in range(n):
    uid = user_ids[i]
    vid = video_ids[i]
    aid = author_ids[i]
    tid = tab_ids[i]
    lv = long_views[i]
    clk = is_clicks[i]
    t = time_ms[i]

    user_hist_rate[i], user_hist_count[i] = get_decayed_stats(user_stats, uid, t)
    video_hist_rate[i], video_hist_count[i] = get_decayed_stats(video_stats, vid, t)
    author_hist_rate[i], author_hist_count[i] = get_decayed_stats(author_stats, aid, t)
    tab_hist_rate[i], tab_hist_count[i] = get_decayed_stats(tab_stats, tid, t)

    uv_key = (uid, vid)
    user_video_hist_rate[i], user_video_hist_count[i] = get_decayed_stats(
        user_video_stats, uv_key, t
    )

    ua_key = (uid, aid)
    user_author_hist_rate[i], user_author_hist_count[i] = get_decayed_stats(
        user_author_stats, ua_key, t
    )

    ut_key = (uid, tid)
    user_tab_hist_rate[i], user_tab_hist_count[i] = get_decayed_stats(
        user_tab_stats, ut_key, t
    )

    # Click-based stats
    user_click_rate[i], user_click_count[i] = get_decayed_stats(
        user_click_stats, uid, t
    )
    video_click_rate[i], video_click_count[i] = get_decayed_stats(
        video_click_stats, vid, t
    )
    author_click_rate[i], author_click_count[i] = get_decayed_stats(
        author_click_stats, aid, t
    )
    user_video_click_rate[i], user_video_click_count[i] = get_decayed_stats(
        user_video_click_stats, uv_key, t
    )

    # Recent window features
    if uid not in user_recent_buffer:
        user_recent_buffer[uid] = []
    recent_list = user_recent_buffer[uid]
    if len(recent_list) >= 20:
        recent_list = recent_list[-19:]
    if len(recent_list) > 0:
        recent_pos_rate[i] = np.mean([x[1] for x in recent_list])
        recent_click_rate[i] = np.mean([x[2] for x in recent_list])
        recent_activity_gap[i] = (t - recent_list[-1][0]) / 1000.0
    else:
        recent_pos_rate[i] = 0.0
        recent_click_rate[i] = 0.0
        recent_activity_gap[i] = 0.0

    author_popularity[i] = author_exposure.get(aid, 0)
    user_activity_count[i] = user_stats.get(uid, [0, 0, 0, 0])[2]
    video_exposure_total[i] = video_stats.get(vid, [0, 0, 0, 0])[2]

    if vid not in video_first_seen:
        video_first_seen[vid] = t
    video_age_ms[i] = t - video_first_seen[vid]

    # Update all stats
    update_stats(user_stats, uid, t, lv)
    update_stats(video_stats, vid, t, lv)
    update_stats(author_stats, aid, t, lv)
    update_stats(tab_stats, tid, t, lv)
    update_stats(user_video_stats, uv_key, t, lv)
    update_stats(user_author_stats, ua_key, t, lv)
    update_stats(user_tab_stats, ut_key, t, lv)

    # Update click stats
    update_stats(user_click_stats, uid, t, clk)
    update_stats(video_click_stats, vid, t, clk)
    update_stats(author_click_stats, aid, t, clk)
    update_stats(user_video_click_stats, uv_key, t, clk)

    author_exposure[aid] = author_exposure.get(aid, 0) + 1

    # Update recent buffer
    user_recent_buffer[uid] = recent_list + [(t, lv, clk)]

all_df["user_hist_rate"] = user_hist_rate
all_df["user_hist_count"] = user_hist_count
all_df["video_hist_rate"] = video_hist_rate
all_df["video_hist_count"] = video_hist_count
all_df["author_hist_rate"] = author_hist_rate
all_df["author_hist_count"] = author_hist_count
all_df["tab_hist_rate"] = tab_hist_rate
all_df["tab_hist_count"] = tab_hist_count
all_df["user_video_hist_rate"] = user_video_hist_rate
all_df["user_video_hist_count"] = user_video_hist_count
all_df["user_author_hist_rate"] = user_author_hist_rate
all_df["user_author_hist_count"] = user_author_hist_count
all_df["user_tab_hist_rate"] = user_tab_hist_rate
all_df["user_tab_hist_count"] = user_tab_hist_count

all_df["user_click_rate"] = user_click_rate
all_df["user_click_count"] = user_click_count
all_df["video_click_rate"] = video_click_rate
all_df["video_click_count"] = video_click_count
all_df["author_click_rate"] = author_click_rate
all_df["author_click_count"] = author_click_count
all_df["user_video_click_rate"] = user_video_click_rate
all_df["user_video_click_count"] = user_video_click_count

all_df["recent_pos_rate"] = recent_pos_rate
all_df["recent_click_rate"] = recent_click_rate
all_df["recent_activity_gap"] = recent_activity_gap

all_df["video_duration_bucket"] = video_duration_bucket
all_df["author_popularity"] = author_popularity
all_df["video_age_ms"] = video_age_ms
all_df["user_activity_count"] = user_activity_count
all_df["video_exposure_total"] = video_exposure_total

all_df["hour"] = (all_df["time_ms"] // 3600000) % 24
all_df["is_weekend"] = ((all_df["date"] % 100) % 7 >= 5).astype(np.int8)
all_df["log_duration"] = np.log1p(all_df["duration_ms"])
all_df["log_video_age"] = np.log1p(all_df["video_age_ms"])
all_df["log_author_popularity"] = np.log1p(all_df["author_popularity"])
all_df["log_user_activity"] = np.log1p(all_df["user_activity_count"])
all_df["log_video_exposure"] = np.log1p(all_df["video_exposure_total"])
all_df["log_recent_activity_gap"] = np.log1p(all_df["recent_activity_gap"])

# Feature interactions
all_df["user_video_rate_diff"] = (
    all_df["user_video_hist_rate"] - all_df["video_hist_rate"]
)
all_df["user_author_rate_diff"] = (
    all_df["user_author_hist_rate"] - all_df["author_hist_rate"]
)
all_df["video_author_rate_diff"] = (
    all_df["video_hist_rate"] - all_df["author_hist_rate"]
)
all_df["user_video_count_ratio"] = (all_df["user_video_hist_count"] + 1) / (
    all_df["video_hist_count"] + 1
)
all_df["user_author_count_ratio"] = (all_df["user_author_hist_count"] + 1) / (
    all_df["author_hist_count"] + 1
)

# Click rate interactions
all_df["user_video_click_diff"] = (
    all_df["user_video_click_rate"] - all_df["video_click_rate"]
)
all_df["video_click_longview_diff"] = (
    all_df["video_click_rate"] - all_df["video_hist_rate"]
)
all_df["user_click_longview_diff"] = (
    all_df["user_click_rate"] - all_df["user_hist_rate"]
)

# Rank-normalized historical rates within each user
all_df["user_video_rate_rank"] = all_df.groupby("user_id")["user_video_hist_rate"].rank(
    pct=True
)
all_df["video_rate_rank"] = all_df.groupby("user_id")["video_hist_rate"].rank(pct=True)
all_df["author_rate_rank"] = all_df.groupby("user_id")["author_hist_rate"].rank(
    pct=True
)
all_df["video_age_rank"] = all_df.groupby("user_id")["video_age_ms"].rank(pct=True)
all_df["video_click_rate_rank"] = all_df.groupby("user_id")["video_click_rate"].rank(
    pct=True
)

# Time-based features
all_df["time_since_last_user_activity"] = (
    all_df.groupby("user_id")["time_ms"].diff() / 1000
)
all_df["time_since_last_user_activity"] = all_df[
    "time_since_last_user_activity"
].fillna(0)
all_df["log_time_since_last"] = np.log1p(all_df["time_since_last_user_activity"])

# User average duration seen so far (past only)
all_df["user_avg_duration_past"] = all_df.groupby("user_id")["duration_ms"].transform(
    lambda x: x.expanding().mean().shift(1)
)
all_df["user_avg_duration_past"] = all_df["user_avg_duration_past"].fillna(
    all_df["duration_ms"]
)
all_df["duration_vs_user_avg"] = all_df["duration_ms"] / (
    all_df["user_avg_duration_past"] + 1
)
all_df["log_duration_vs_user_avg"] = np.log1p(all_df["duration_vs_user_avg"])

train_feat = all_df[all_df["split"] == "train"].copy()
valid_feat = all_df[all_df["split"] == "valid"].copy()
test_feat = all_df[all_df["split"] == "test"].copy()
# Restore official file-order (data.load()["test"]) for row_id alignment
test_feat = test_feat.sort_values("_orig_idx").reset_index(drop=True)

print(
    f"Features computed. Train: {len(train_feat)}, Valid: {len(valid_feat)}, Test: {len(test_feat)}",
    flush=True,
)

feature_cols = [
    "duration_ms",
    "log_duration",
    "hour",
    "is_weekend",
    "user_hist_rate",
    "user_hist_count",
    "video_hist_rate",
    "video_hist_count",
    "author_hist_rate",
    "author_hist_count",
    "tab_hist_rate",
    "tab_hist_count",
    "user_video_hist_rate",
    "user_video_hist_count",
    "user_author_hist_rate",
    "user_author_hist_count",
    "user_tab_hist_rate",
    "user_tab_hist_count",
    "user_click_rate",
    "user_click_count",
    "video_click_rate",
    "video_click_count",
    "author_click_rate",
    "author_click_count",
    "user_video_click_rate",
    "user_video_click_count",
    "recent_pos_rate",
    "recent_click_rate",
    "recent_activity_gap",
    "log_recent_activity_gap",
    "video_duration_bucket",
    "author_popularity",
    "video_age_ms",
    "log_video_age",
    "log_author_popularity",
    "log_user_activity",
    "log_video_exposure",
    "user_video_rate_rank",
    "video_rate_rank",
    "author_rate_rank",
    "video_age_rank",
    "video_click_rate_rank",
    "user_video_rate_diff",
    "user_author_rate_diff",
    "video_author_rate_diff",
    "user_video_count_ratio",
    "user_author_count_ratio",
    "user_video_click_diff",
    "video_click_longview_diff",
    "user_click_longview_diff",
    "time_since_last_user_activity",
    "log_time_since_last",
    "duration_vs_user_avg",
    "log_duration_vs_user_avg",
    "tab",
]

train_feat = train_feat.sort_values(["user_id", "time_ms"])
train_capped = (
    train_feat.groupby("user_id", group_keys=False).tail(10000).reset_index(drop=True)
)
train_capped = train_capped.sort_values(["user_id", "time_ms"]).reset_index(drop=True)

valid_feat_sorted = valid_feat.sort_values(["user_id", "time_ms"])
valid_capped = (
    valid_feat_sorted.groupby("user_id", group_keys=False)
    .tail(10000)
    .reset_index(drop=True)
)
valid_capped = valid_capped.sort_values(["user_id", "time_ms"]).reset_index(drop=True)

X_train = train_capped[feature_cols].values
y_train = train_capped["long_view"].values
groups_train = train_capped.groupby("user_id", sort=False).size().values

X_valid_es = valid_capped[feature_cols].values
y_valid_es = valid_capped["long_view"].values
groups_valid_es = valid_capped.groupby("user_id", sort=False).size().values

print(
    f"Training rows after capping: {len(train_capped)}, users: {len(groups_train)}",
    flush=True,
)
print(
    f"Validation rows for early stopping: {len(valid_capped)}, users: {len(groups_valid_es)}",
    flush=True,
)

lgb_train = lgb.Dataset(X_train, label=y_train, group=groups_train)
lgb_valid = lgb.Dataset(
    X_valid_es, label=y_valid_es, group=groups_valid_es, reference=lgb_train
)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": os.cpu_count(),
}

print("Training LightGBM LambdaRank with early stopping...", flush=True)
t1 = time.time()
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_valid],
    callbacks=[lgb.early_stopping(100, verbose=True)],
)
print(
    f"Training completed in {time.time()-t1:.1f}s, best_iter={model.best_iteration}",
    flush=True,
)

X_valid = valid_feat[feature_cols].values
y_valid = valid_feat["long_view"].values
u_valid = valid_feat["user_id"].values

scores_valid = model.predict(X_valid, num_iteration=model.best_iteration)

r = evaluate(u_valid, y_valid, scores_valid)
print(f'Validation GAUC: {r["GAUC"]:.4f}')
print(f'Validation nDCG@5: {r["nDCG@5"]:.4f}')
print(f'Validation primary: {r["primary"]:.4f}')

X_test = test_feat[feature_cols].values
scores_test = model.predict(X_test, num_iteration=model.best_iteration)

os.makedirs("submission", exist_ok=True)
sub_df = pd.DataFrame(
    {
        "row_id": np.arange(len(test_feat)),
        "user_id": test_feat["user_id"].values,
        "video_id": test_feat["video_id"].values,
        "score": scores_test,
    }
)
sub_df.to_csv("submission/submission.csv", index=False)
print(
    f"Submission saved to submission/submission.csv with {len(sub_df)} rows", flush=True
)
