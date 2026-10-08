import argparse, os, sys, json, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
from models.islr_model import build_islr_model

class AttnPool(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * dim ** -0.5)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, mask):
        q = self.q.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=~mask)
        return self.norm(out.squeeze(1))

class ArcFace(nn.Module):
    def __init__(self, dim, n_cls, s=30.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, dim)); nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m
    def forward(self, feat, labels=None):
        return F.linear(F.normalize(feat), F.normalize(self.W)) * self.s

class Head(nn.Module):
    def __init__(self, dim, n_cls, pool="attn", arcface=False, dropout=0.3):
        super().__init__()
        self.pool = AttnPool(dim) if pool == "attn" else None
        self.drop = nn.Dropout(dropout)
        self.arcface = ArcFace(dim, n_cls) if arcface else None
        self.fc = None if arcface else nn.Linear(dim, n_cls)
    def forward(self, x, mask, labels=None):
        z = self.pool(x, mask) if self.pool is not None else (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        z = self.drop(z)
        return self.arcface(z, labels) if self.arcface is not None else self.fc(z)

def corrupt_pose(p, ctype, sev, rng):
    out = p.copy(); T = out.shape[0]
    if sev <= 0: return out
    if ctype == 'dropout':
        m = rng.random(T) < sev; out[m] = 0.0
    elif ctype == 'hand_occl':
        m = rng.random(T) < sev
        out[np.ix_(np.where(m)[0], list(range(132, 258)))] = 0.0
        out[np.ix_(np.where(m)[0], [259, 260])] = 0.0
    elif ctype == 'noise':
        coord = list(range(0, 258))
        s = sev * (np.abs(out[:, coord]).mean() + 1e-6)
        out[:, coord] += rng.normal(0, s, out[:, coord].shape).astype(np.float32)
    return out

def read_csv(path, vcol='Video file', gcol='Gloss'):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f); cols = {c.lower().strip(): c for c in r.fieldnames}
        vc, gc = cols.get(vcol.lower(), vcol), cols.get(gcol.lower(), gcol)
        for x in r:
            rows.append((os.path.splitext(os.path.basename(x[vc].strip()))[0], x[gc].strip()))
    return rows

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose_ckpt', required=True); ap.add_argument('--pose_vocab', required=True)
    ap.add_argument('--pose_feat_dir', required=True)
    ap.add_argument('--signrep_head', required=True); ap.add_argument('--signrep_feat_dir', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--ctype', default='dropout'); ap.add_argument('--sevs', type=float, nargs='+', default=[0.0,0.1,0.25,0.5,0.75,0.9])
    ap.add_argument('--alpha', type=float, default=0.5); ap.add_argument('--max_seg', type=int, default=64)
    ap.add_argument('--out_dir', default='results_islr_degrade')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda'); rng = np.random.default_rng(0)

    pose_itos = json.load(open(args.pose_vocab)); pose_stoi = {g: i for i, g in enumerate(pose_itos)}
    pose = build_islr_model(num_classes=len(pose_itos), input_dim=261, size='base', dropout=0.0).to(dev).eval()
    pose.load_state_dict(torch.load(args.pose_ckpt, map_location=dev)['model_state'])

    ck = torch.load(args.signrep_head, map_location=dev); sa = ck['args']
    head = Head(768, len(ck['itos']), pool=sa['pool'], arcface=sa['arcface'], dropout=0.0).to(dev).eval()
    head.load_state_dict(ck['model_state']); sr_stoi = ck['stoi']
    proj = np.full(len(pose_itos), -1)
    for g, i in sr_stoi.items():
        if g in pose_stoi: proj[pose_stoi[g]] = i

    rows = read_csv(args.test_csv)
    items = []
    for vid, g in rows:
        if g not in pose_stoi: continue
        pp = os.path.join(args.pose_feat_dir, vid + '.pt'); sp = os.path.join(args.signrep_feat_dir, vid + '.npy')
        if os.path.exists(pp) and os.path.exists(sp):
            items.append((vid, pose_stoi[g], pp, sp))
    print(f'test clips usable: {len(items)}', flush=True)

    NEG = -1e4
    results = []
    for sev in args.sevs:
        n = 0; pose_ok = sr_ok = fus_ok = 0
        rels, fc_l, sc_l = [], [], []
        for vid, y, pp, sp in items:
            pfeat = torch.load(pp, map_location='cpu').float().numpy()
            pc = corrupt_pose(pfeat, args.ctype, sev, rng)
            x = torch.from_numpy(pc).float().unsqueeze(0).to(dev)
            lp = pose(x, torch.tensor([pc.shape[0]], device=dev), labels=None)['logits'][0]
            lp = F.log_softmax(lp, -1)
            sf = np.load(sp).astype(np.float32)
            if sf.shape[0] > args.max_seg: sf = sf[(sf.shape[0]-args.max_seg)//2:][:args.max_seg]
            sx = torch.from_numpy(sf).unsqueeze(0).to(dev); sm = torch.ones(1, sf.shape[0], dtype=torch.bool, device=dev)
            sl = F.log_softmax(head(sx, sm), -1)[0]
            sl_canon = torch.full((len(pose_itos),), NEG, device=dev)
            valid = proj >= 0
            sl_canon[torch.from_numpy(np.where(valid)[0]).to(dev)] = sl[torch.from_numpy(proj[valid]).to(dev)]
            fused = (1 - args.alpha) * lp + args.alpha * sl_canon
            pose_ok += int(lp.argmax().item() == y)
            sr_ok += int(sl_canon.argmax().item() == y)
            fus_ok += int(fused.argmax().item() == y)
            rels.append(float(pc[:, 258:261].mean()))
            fc_l.append(int(fused.argmax().item() == y)); sc_l.append(int(sl_canon.argmax().item() == y))
            n += 1
        rels = np.array(rels); fc = np.array(fc_l); sc = np.array(sc_l)
        best_sw = -1
        for thr in np.quantile(rels, np.linspace(0, 1, 21)):
            acc = np.where(rels < thr, sc, fc).mean()
            best_sw = max(best_sw, acc)
        row = dict(severity=sev, pose_only=round(100*pose_ok/n,2), signrep_only=round(100*sr_ok/n,2),
                   fusion=round(100*fus_ok/n,2), reliability_switch=round(100*best_sw,2))
        results.append(row)
        print(f'[{args.ctype} sev={sev}] pose={row["pose_only"]} signrep={row["signrep_only"]} fusion={row["fusion"]} switch={row["reliability_switch"]}', flush=True)

    with open(os.path.join(args.out_dir, f'islr_degrade_{args.ctype}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    json.dump(results, open(os.path.join(args.out_dir, f'islr_degrade_{args.ctype}.json'), 'w'), indent=2)
    print('RESULT:', json.dumps(results, indent=2), flush=True)

if __name__ == '__main__':
    main()
