"""
DIN 序列建模参考实现（已验证 val primary 0.6005 / test 0.5940，≈ 官方 FM baseline）。

这是给 agent 的【参考起点】，不是锁死的答案。看懂核心思路后可以：
- 加多任务头（is_click/is_like 辅助）
- 历史序列加行为信号 / GRU(DIEN) 演化
- 加连续特征、换更强结构

核心三点：
1. 历史序列：按 time_ms 排序 + 去重 + 最近 ~50 个（真正的序列，比 groupby 统计强）。
2. target attention (DIN)：用候选视频当 query 去 attend 历史。
3. 排序 loss 必须（listwise softmax / BPR）：BCE 在这里只有 0.478（接近随机）。

注意：data.load() 丢弃了 time_ms，所以这里直接用 pandas 读原始日志构造序列。
"""
import sys, os, time
from collections import defaultdict
sys.path.insert(0, "input")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import evaluate

MAX_LEN = 50
EMB = 32
EPOCHS = 3
LR = 3e-4
SEED = 42
DATA = "input"

np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"device: {device}", flush=True)

# ---------- 1. 读原始日志（需要 time_ms 构造序列） ----------
cols = ["user_id", "video_id", "date", "time_ms", "long_view", "tab", "duration_ms"]
tr = pd.read_csv(f"{DATA}/log_standard_4_08_to_4_21_pure.csv", usecols=cols)
va = pd.read_csv(f"{DATA}/log_standard_4_22_to_5_08_pure.csv", usecols=cols)
vb = pd.read_csv(f"{DATA}/video_features_basic_pure.csv", usecols=["video_id", "author_id"])
vid2author = dict(zip(vb.video_id.astype(str), vb.author_id.astype(str)))
for df in (tr, va):
    df["author_id"] = df.video_id.astype(str).map(vid2author).fillna("UNK")

train = tr[tr.date <= 20220421].reset_index(drop=True)
valid = va[(va.date >= 20220422) & (va.date <= 20220428)].reset_index(drop=True)
test = va[va.date >= 20220429].reset_index(drop=True)

# ---------- 2. 编码（5 字段） ----------
def build_vocab(vals):
    v = {}
    for x in vals:
        if x not in v: v[x] = len(v)
    return v

edges = np.quantile(train.duration_ms.values, np.linspace(0, 1, 11)[1:-1])
def durbucket(d): return str(int(np.searchsorted(edges, d)))

vocab_u = build_vocab(train.user_id.astype(str).tolist()); vocab_u["<UNK>"] = len(vocab_u)
vocab_v = build_vocab(train.video_id.astype(str).tolist()); vocab_v["<UNK>"] = len(vocab_v)
vocab_a = build_vocab(train.author_id.tolist()); vocab_a["<UNK>"] = len(vocab_a)
vocab_t = build_vocab(train.tab.astype(str).tolist()); vocab_t["<UNK>"] = len(vocab_t)
vocab_d = build_vocab([durbucket(d) for d in train.duration_ms]); vocab_d["<UNK>"] = len(vocab_d)
N_U, N_V, N_A, N_T, N_D = len(vocab_u), len(vocab_v), len(vocab_a), len(vocab_t), len(vocab_d)

def enc(series, vocab, tostr=False):
    if tostr:
        return np.array([vocab.get(str(x), vocab["<UNK>"]) for x in series], dtype=np.int64)
    return np.array([vocab.get(x, vocab["<UNK>"]) for x in series], dtype=np.int64)

def encode_df(df):
    return {"user": enc(df.user_id, vocab_u, True), "video": enc(df.video_id, vocab_v, True),
            "author": enc(df.author_id, vocab_a), "tab": enc(df.tab, vocab_t, True),
            "dur": enc([durbucket(d) for d in df.duration_ms], vocab_d),
            "y": df.long_view.values.astype(np.float32)}

tr_enc = encode_df(train)
va_enc = encode_df(valid)
te_enc = encode_df(test)

# ---------- 3. 历史序列（time_ms 排序 + 去重 + 最近 MAX_LEN） ----------
def recent_unique(seq, max_len):
    seen, out = set(), []
    for v in reversed(seq):
        if v not in seen:
            seen.add(v); out.append(v)
            if len(out) >= max_len: break
    return out[::-1]

ts = train.assign(_u=train.user_id.astype(str), _v=tr_enc["video"],
                  _t=train.time_ms).sort_values(["_u", "_t"], kind="stable")
user_hist = {}
for u, g in ts.groupby("_u", sort=False):
    user_hist[u] = np.array(recent_unique(g._v.values, MAX_LEN), dtype=np.int64)

hist_mat = np.zeros((N_U, MAX_LEN), dtype=np.int64)
hist_mask = np.zeros((N_U, MAX_LEN), dtype=np.float32)
for u, seq in user_hist.items():
    uid = vocab_u[u]; m = len(seq)
    hist_mat[uid, :m] = seq; hist_mask[uid, :m] = 1.0

# 按用户分组的曝光行索引（listwise 用）
user_to_rows = defaultdict(list)
for i, u in enumerate(tr_enc["user"]):
    user_to_rows[u].append(i)
user_rows_list = [(u, np.array(r, dtype=np.int64)) for u, r in user_to_rows.items()]

# ---------- 4. DIN 模型 ----------
class DIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_emb = nn.Embedding(N_U, EMB)
        self.video_emb = nn.Embedding(N_V, EMB)
        self.author_emb = nn.Embedding(N_A, EMB)
        self.tab_emb = nn.Embedding(N_T, EMB)
        self.dur_emb = nn.Embedding(N_D, EMB)
        for m in [self.user_emb, self.video_emb, self.author_emb, self.tab_emb, self.dur_emb]:
            nn.init.xavier_uniform_(m.weight)
        self.att = nn.Sequential(nn.Linear(EMB*3, 32), nn.ReLU(), nn.Linear(32, 1))
        self.mlp = nn.Sequential(
            nn.Linear(EMB*3, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, user, video, author, tab, dur, hist, hmask):
        u = self.user_emb(user)
        t = self.video_emb(video) + self.author_emb(author) + self.tab_emb(tab) + self.dur_emb(dur)
        h = self.video_emb(hist)
        L = h.size(1)
        t_exp = t.unsqueeze(1).expand(-1, L, -1)
        aw = self.att(torch.cat([h, t_exp, h * t_exp], dim=-1)).squeeze(-1)
        mask = hmask * (hist != video.unsqueeze(1)).float()
        aw = aw.masked_fill(mask == 0, -1e9)
        aw = torch.softmax(aw, dim=1).unsqueeze(-1)
        interest = (aw * h).sum(dim=1)
        return self.mlp(torch.cat([u, t, interest], dim=-1)).squeeze(-1)

model = DIN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

def to_t(x): return torch.from_numpy(x).to(device)
E_U, E_V, E_A = to_t(tr_enc["user"]), to_t(tr_enc["video"]), to_t(tr_enc["author"])
E_T, E_D, E_Y = to_t(tr_enc["tab"]), to_t(tr_enc["dur"]), to_t(tr_enc["y"])
H, HM = to_t(hist_mat), to_t(hist_mask)

# ---------- 5. 训练（listwise softmax + BPR，按用户） ----------
best_primary, best_state, patience = 0.0, None, 0
for epoch in range(EPOCHS):
    model.train()
    perm = np.random.permutation(len(user_rows_list))
    total, cnt = 0.0, 0
    for bi in range(0, len(user_rows_list), 32):
        batch = [user_rows_list[i] for i in perm[bi:bi+32]]
        rows = np.concatenate([r for _, r in batch])
        u, v, a = E_U[rows], E_V[rows], E_A[rows]
        t, d, y = E_T[rows], E_D[rows], E_Y[rows]
        scores = model(u, v, a, t, d, H[u], HM[u])
        loss = torch.tensor(0.0, device=device); c = 0
        off = 0
        for _, r in batch:
            k = len(r); sc = scores[off:off+k]; yy = y[off:off+k]
            pos = yy > 0.5; npos = pos.sum()
            if 0 < npos < k:
                soft = torch.log_softmax(sc, dim=0)
                target = pos.float() / npos.float()
                loss = loss - (target * soft).sum()
                pi = pos.nonzero().squeeze(-1); ni = (~pos).nonzero().squeeze(-1)
                diff = sc[pi].unsqueeze(1) - sc[ni].unsqueeze(0)
                loss = loss + 0.5 * (-torch.log(torch.sigmoid(diff) + 1e-8)).mean()
                c += 1
            off += k
        if c > 0:
            loss = loss / c
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); cnt += 1

    # 评估 valid
    model.eval()
    with torch.no_grad():
        u = to_t(va_enc["user"]); v = to_t(va_enc["video"]); a = to_t(va_enc["author"])
        t = to_t(va_enc["tab"]); d = to_t(va_enc["dur"])
        outs = []
        for i in range(0, len(u), 2048):
            uid = u[i:i+2048]
            outs.append(model(uid, v[i:i+2048], a[i:i+2048], t[i:i+2048],
                              d[i:i+2048], H[uid], HM[uid]).cpu().numpy())
        sv = np.concatenate(outs)
    r = evaluate(valid.user_id.astype(str).tolist(), va_enc["y"], sv)
    print(f"Epoch {epoch+1} | val GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} primary {r['primary']:.4f}", flush=True)
    if r["primary"] > best_primary + 1e-4:
        best_primary = r["primary"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1
        if patience >= 2:
            break

model.load_state_dict(best_state); model.to(device)

# ---------- 6. 生成 submission ----------
model.eval()
with torch.no_grad():
    u = to_t(te_enc["user"]); v = to_t(te_enc["video"]); a = to_t(te_enc["author"])
    t = to_t(te_enc["tab"]); d = to_t(te_enc["dur"])
    outs = []
    for i in range(0, len(u), 2048):
        uid = u[i:i+2048]
        outs.append(model(uid, v[i:i+2048], a[i:i+2048], t[i:i+2048],
                          d[i:i+2048], H[uid], HM[uid]).cpu().numpy())
    st = np.concatenate(outs)

os.makedirs("submission", exist_ok=True)
rows = []
for i, (row, score) in enumerate(zip(test.itertuples(index=False), st)):
    rows.append((i, row.user_id, row.video_id, float(score)))
pd.DataFrame(rows, columns=["row_id", "user_id", "video_id", "score"]).to_csv(
    "submission/submission.csv", index=False)
print(f"Validation primary: {best_primary:.4f}", flush=True)
print("Submission saved.", flush=True)
