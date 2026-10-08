import os, csv, json, math, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def read_split(csv_path, video_col, gloss_col):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in reader.fieldnames}
        vcol = cols.get(video_col.lower(), video_col)
        gcol = cols.get(gloss_col.lower(), gloss_col)
        for r in reader:
            vid = os.path.splitext(os.path.basename(r[vcol].strip()))[0]
            rows.append((vid, r[gcol].strip()))
    return rows


def build_vocab(rows):
    glosses = sorted({g for _, g in rows})
    stoi = {g: i for i, g in enumerate(glosses)}
    return stoi, glosses


class FeatDataset(Dataset):
    def __init__(self, rows, feat_dir, stoi, max_seg=64, train=False):
        self.feat_dir = feat_dir
        self.stoi = stoi
        self.max_seg = max_seg
        self.train = train
        self.items = []
        miss = 0
        for vid, g in rows:
            if g not in stoi:
                continue
            p = os.path.join(feat_dir, vid + ".npy")
            if os.path.exists(p):
                self.items.append((p, stoi[g]))
            else:
                miss += 1
        if miss:
            print(f"  [warn] {miss} clips missing features in {feat_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        x = np.load(p).astype(np.float32)
        if x.shape[0] == 0:
            x = np.zeros((1, 768), np.float32)
        n = x.shape[0]
        if n > self.max_seg:
            if self.train:
                s = random.randint(0, n - self.max_seg)
            else:
                s = (n - self.max_seg) // 2
            x = x[s:s + self.max_seg]
        return torch.from_numpy(x), y


def collate(batch):
    xs, ys = zip(*batch)
    lens = torch.tensor([x.shape[0] for x in xs])
    T = int(lens.max())
    D = xs[0].shape[1]
    out = torch.zeros(len(xs), T, D)
    mask = torch.zeros(len(xs), T, dtype=torch.bool)
    for i, x in enumerate(xs):
        out[i, :x.shape[0]] = x
        mask[i, :x.shape[0]] = True
    return out, mask, torch.tensor(ys)


class AttnPool(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * dim ** -0.5)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        q = self.q.expand(x.shape[0], -1, -1)
        key_pad = ~mask
        out, _ = self.attn(q, x, x, key_padding_mask=key_pad)
        return self.norm(out.squeeze(1))


class ArcFace(nn.Module):
    def __init__(self, dim, n_cls, s=30.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, feat, labels=None):
        cos = F.linear(F.normalize(feat), F.normalize(self.W))
        if labels is None:
            return cos * self.s
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, cos.shape[1]).bool()
        return torch.where(onehot, target, cos) * self.s


class Head(nn.Module):
    def __init__(self, dim, n_cls, pool="attn", arcface=False, dropout=0.3):
        super().__init__()
        self.pool = AttnPool(dim) if pool == "attn" else None
        self.drop = nn.Dropout(dropout)
        self.arcface = ArcFace(dim, n_cls) if arcface else None
        self.fc = None if arcface else nn.Linear(dim, n_cls)

    def forward(self, x, mask, labels=None):
        if self.pool is not None:
            z = self.pool(x, mask)
        else:
            z = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        z = self.drop(z)
        if self.arcface is not None:
            return self.arcface(z, labels)
        return self.fc(z)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    r1 = r5 = n = 0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask, labels=None)
        top5 = logits.topk(5, dim=-1).indices
        r1 += (top5[:, 0] == y).sum().item()
        r5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
        n += y.numel()
    return 100.0 * r1 / n, 100.0 * r5 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--video_col", default="Video file")
    ap.add_argument("--gloss_col", default="Gloss")
    ap.add_argument("--pool", choices=["attn", "mean"], default="attn")
    ap.add_argument("--arcface", action="store_true")
    ap.add_argument("--max_seg", type=int, default=64)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label_smooth", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = read_split(args.train_csv, args.video_col, args.gloss_col)
    val_rows = read_split(args.val_csv, args.video_col, args.gloss_col)
    test_rows = read_split(args.test_csv, args.video_col, args.gloss_col)
    stoi, itos = build_vocab(train_rows)
    print(f"vocab={len(itos)} glosses | train={len(train_rows)} "
          f"val={len(val_rows)} test={len(test_rows)}")

    def mk(rows, train):
        ds = FeatDataset(rows, args.feat_dir, stoi, args.max_seg, train)
        return DataLoader(ds, batch_size=args.bs, shuffle=train,
                          num_workers=args.workers, collate_fn=collate,
                          pin_memory=True, drop_last=train)
    train_loader = mk(train_rows, True)
    val_loader = mk(val_rows, False)
    test_loader = mk(test_rows, False)
    print(f"usable: train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = Head(768, len(itos), pool=args.pool,
                 arcface=args.arcface, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask, labels=y if args.arcface else None)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * y.numel()
        sched.step()
        v1, v5 = evaluate(model, val_loader, device)
        print(f"ep {ep:3d} | loss {tot/len(train_loader.dataset):.4f} "
              f"| val R@1 {v1:.2f} R@5 {v5:.2f}", flush=True)
        if v1 > best:
            best = v1
            torch.save({"model_state": model.state_dict(),
                        "stoi": stoi, "itos": itos, "args": vars(args)},
                       os.path.join(args.out_dir, "best.pt"))

    ck = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ck["model_state"])
    t1, t5 = evaluate(model, test_loader, device)
    print(f"\nBEST val R@1 {best:.2f} | TEST R@1 {t1:.2f} R@5 {t5:.2f}")
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump({"val_r1": best, "test_r1": t1, "test_r5": t5}, f, indent=2)


if __name__ == "__main__":
    main()
