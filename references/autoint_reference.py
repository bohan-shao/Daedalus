#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KuaiRand-Pure 最强 baseline：AutoInt(AFI) + 二值标签
=====================================================
任务：用户内排序（within-user ranking over logged impressions），标签 long_view。
指标：GAUC + nDCG@5，primary = 两者平均（见 ./input/evaluate.py）。

模型：AFI（AutoInt，multi-head self-attention 建模字段交互，torch 实现，参考
D2Co 论文《Uncovering User Interest from Biased and Noised Watch Time in Video
Recommendation》的模型组）。实测 valid primary 0.6032，超过官方 FM baseline
（0.6015），是该任务当前最强的模型。

为什么是这个配置（实测结论，MLEvolve 迭代勿重复踩坑）：
  - 本任务 long_view 本质上是观看时长的阈值函数（≈ play>18s，或短视频看完），
    本身已时长归一化，因此二值标签与指标天然对齐。
  - D2Co 的 GMM watch-time 连续去偏标签（d2co/d2co_lin，以及 PCR/scale_wt/
    long_view2 变体）在 FM 和 AFI 下都低于二值标签（afi+d2co=0.5758），已被裁剪。
  - DFM/NFM/AFM 等模型也低于 AFI（0.588-0.590），已裁剪。
  后续值得迭代的方向：AFI+更多特征（music_id/video_type/upload_type/tag_pop）、
  更长训练/调 atten_dim/heads、pairwise/listwise 损失、用户行为序列。

符合任务契约：
  - 特征只用 user_id/video_id/author_id/tab/dur_bucket（观看时长不进入特征）。
  - valid 指标按官方 evaluate.py 计算并打印：GAUC=.. nDCG@5=.. primary=..
  - 写 ./submission/submission.csv（row_id,user_id,video_id,score）
  - 最后一行打印 Final Validation Score: <primary>
依赖 numpy + torch + 官方 evaluate.py。
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- 布局检测：MLEvolve workspace（./input）或任务本地（./data） ----------------
if os.path.isdir(os.path.join(os.getcwd(), 'input', 'data')):
    DATA_DIR = os.path.join(os.getcwd(), 'input', 'data')
    sys.path.insert(0, os.path.join(os.getcwd(), 'input'))
    SUB_DIR = os.path.join(os.getcwd(), 'submission')
else:
    DATA_DIR = os.path.join(HERE, 'data')
    sys.path.insert(0, HERE)
    SUB_DIR = os.path.join(HERE, 'submission')

from evaluate import evaluate  # noqa: E402  官方指标实现（GAUC / nDCG@5 / primary）

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("需要 torch（MLEvolve 环境已安装）；当前环境缺失 torch")

FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}


# ============================ 1. 数据加载（保持官方行序） ============================
def load(data_dir):
    """读两份 standard 日志 + 视频基础特征，返回按划分切好的行列表。

    每行: (date, user_id, video_id, author_id, tab, duration_ms, play_time_ms, long_view)
    行序：先 log_standard_4_08_to_4_21_pure.csv 再 log_standard_4_22_to_5_08_pure.csv，
    按 date 过滤后保持原文件顺序 —— 与 data.py 的评测集行序一致（submission 对齐用）。
    """
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), float(r['play_time_ms']),
                             1 if r['long_view'] != '0' else 0))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


# ============================ 2. 特征编码（train 建词表 + UNK） ============================
def encode(rows_by_split):
    """类别特征 → 连续 id（train 建词表，未见过取值落 UNK 槽），dur_bucket 由 train 分位数定。"""
    tr = rows_by_split['train']
    dur_edges = np.quantile(np.asarray([x[5] for x in tr]), np.linspace(0, 1, 11)[1:-1])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(dur_edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in rows_by_split.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
        enc[name] = X
    return enc, int(sum(field_dims))


# ============================ 3. AFI 模型（AutoInt） ============================
class AFI(nn.Module):
    """AutoInt：multi-head self-attention 建模字段交互（输出 logits）。"""
    def __init__(self, total_dim, n_fields, k=16, num_heads=4, num_layers=1,
                 atten_dim=32, mlp_dims=(64,), dropout=0.2, has_residual=True):
        super().__init__()
        self.emb = nn.Embedding(total_dim, k)
        self.attn_fc = nn.Linear(k, atten_dim)
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(atten_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)])
        self.has_residual = has_residual
        dims = [n_fields * atten_dim] + list(mlp_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(dims[-1], 1)

    def forward(self, x):
        E = self.emb(x)                                    # (B,F,k)
        h = F.relu(self.attn_fc(E))                        # (B,F,atten_dim)
        for attn in self.self_attns:
            out, _ = attn(h, h, h)
            h = (h + out) if self.has_residual else out
            h = F.relu(h)
        flat = h.reshape(h.size(0), -1)
        return self.fc_out(self.mlp(flat))


def train(model, Xtr, ytr, Xva, uva, yva, epochs, batch, lr, patience, seed, device, t0):
    """Adam + BCEWithLogits，early stop 按 valid primary。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.long)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32).to(device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.long)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        idx = np.random.permutation(len(ytr))
        losses = []
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            out = model(Xtr_t[b].to(device)).squeeze(-1)
            loss = loss_fn(out, ytr_t[b])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(Xva_t.to(device))).squeeze(-1).cpu().numpy()
        va = evaluate(uva, yva, pred)
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
              f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, X, device):
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.long).to(device))).squeeze(-1).cpu().numpy()


# ============================ 4. 主流程 ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--atten_dim', type=int, default=32, help='attention 隐维度')
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    print(f"loading {a.data_dir} ...")
    rows = load(a.data_dir)
    print({k: len(v) for k, v in rows.items()}, f"fields={FIELDS}", flush=True)

    ytr = np.array([r[7] for r in rows['train']], dtype=np.float32)
    yva = np.array([r[7] for r in rows['valid']], dtype=np.float32)
    yte = np.array([r[7] for r in rows['test']], dtype=np.float32)
    uva = [r[1] for r in rows['valid']]
    ute = [r[1] for r in rows['test']]

    enc, dim = encode(rows)
    Xtr, Xva, Xte = enc['train'], enc['valid'], enc['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training AFI on {device} ...")
    model = AFI(dim, len(FIELDS), k=a.k, num_heads=a.heads, num_layers=a.layers, atten_dim=a.atten_dim)
    model = train(model, Xtr, ytr, Xva, uva, yva,
                  a.epochs, a.batch, a.lr, a.patience, a.seed, device, t0)
    sva = predict(model, Xva, device)
    ste = predict(model, Xte, device)

    res_v = evaluate(uva, yva, sva)
    res_t = evaluate(ute, yte, ste)
    print(f"\n=== AFI baseline seed={a.seed} ===")
    print(f"  valid  GAUC {res_v['GAUC']:.4f} | nDCG@5 {res_v['nDCG@5']:.4f} | primary {res_v['primary']:.4f}")
    print(f"  test   GAUC {res_t['GAUC']:.4f} | nDCG@5 {res_t['nDCG@5']:.4f} | primary {res_t['primary']:.4f}")

    # 写 submission（与评测集 test 行序严格对齐）
    os.makedirs(SUB_DIR, exist_ok=True)
    with open(os.path.join(SUB_DIR, 'submission.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, (r, s) in enumerate(zip(rows['test'], ste)):
            w.writerow([i, r[1], r[2], f"{s:.6f}"])
    print(f"wrote {os.path.join(SUB_DIR, 'submission.csv')} ({len(ste)} rows)", flush=True)

    print(f"GAUC={res_v['GAUC']:.4f} nDCG@5={res_v['nDCG@5']:.4f} primary={res_v['primary']:.4f}")
    print(f"Final Validation Score: {res_v['primary']}")


if __name__ == '__main__':
    main()
