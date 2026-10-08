import argparse, os, sys, json, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
from models.islr_model import build_islr_model


class AttnPool(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__(); self.q = nn.Parameter(torch.randn(1, 1, dim) * dim ** -0.5)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True); self.norm = nn.LayerNorm(dim)
    def forward(self, x, mask):
        out, _ = self.attn(self.q.expand(x.shape[0], -1, -1), x, x, key_padding_mask=~mask)
        return self.norm(out.squeeze(1))
class ArcFace(nn.Module):
    def __init__(self, dim, n, s=30.0, m=0.30):
        super().__init__(); self.W = nn.Parameter(torch.randn(n, dim)); nn.init.xavier_uniform_(self.W); self.s = s
    def forward(self, f, labels=None): return F.linear(F.normalize(f), F.normalize(self.W)) * self.s
class Head(nn.Module):
    def __init__(self, dim, n, pool="attn", arcface=False, dropout=0.3):
        super().__init__(); self.pool = AttnPool(dim) if pool == "attn" else None
        self.drop = nn.Dropout(dropout); self.arcface = ArcFace(dim, n) if arcface else None
        self.fc = None if arcface else nn.Linear(dim, n)
    def forward(self, x, mask, labels=None):
        z = self.pool(x, mask) if self.pool is not None else (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        z = self.drop(z); return self.arcface(z) if self.arcface is not None else self.fc(z)


def corrupt_pose(p, ctype, sev, rng):
    out = p.copy(); T = out.shape[0]
    if sev <= 0: return out
    if ctype == 'dropout':
        out[rng.random(T) < sev] = 0.0
    elif ctype == 'hand_occl':
        m = rng.random(T) < sev
        out[np.ix_(np.where(m)[0], list(range(132, 258)))] = 0.0
        out[np.ix_(np.where(m)[0], [259, 260])] = 0.0
    elif ctype == 'noise':
        s = sev * (np.abs(out[:, :258]).mean() + 1e-6)
        out[:, :258] += rng.normal(0, s, out[:, :258].shape).astype(np.float32)
    return out


def read_csv(path, vcol='Video file', gcol='Gloss'):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f); cols = {c.lower().strip(): c for c in r.fieldnames}
        vc, gc = cols.get(vcol.lower(), vcol), cols.get(gcol.lower(), gcol)
        for x in r: rows.append((os.path.splitext(os.path.basename(x[vc].strip()))[0], x[gc].strip()))
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose_ckpt', required=True); ap.add_argument('--pose_vocab', required=True)
    ap.add_argument('--pose_feat_dir', required=True)
    ap.add_argument('--signrep_head', required=True); ap.add_argument('--signrep_feat_dir', required=True)
    ap.add_argument('--val_csv', required=True); ap.add_argument('--test_csv', required=True)
    ap.add_argument('--ctypes', nargs='+', default=['dropout', 'hand_occl'])
    ap.add_argument('--sevs', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--alpha', type=float, default=0.5); ap.add_argument('--max_seg', type=int, default=64)
    ap.add_argument('--out_dir', default='results_islr_final')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); dev = torch.device('cuda')

    itos = json.load(open(args.pose_vocab)); stoi = {g: i for i, g in enumerate(itos)}
    pose = build_islr_model(num_classes=len(itos), input_dim=261, size='base', dropout=0.0).to(dev).eval()
    pose.load_state_dict(torch.load(args.pose_ckpt, map_location=dev)['model_state'])
    ck = torch.load(args.signrep_head, map_location=dev); sa = ck['args']
    head = Head(768, len(ck['itos']), pool=sa['pool'], arcface=sa['arcface'], dropout=0.0).to(dev).eval()
    head.load_state_dict(ck['model_state'])
    proj = np.full(len(itos), -1)
    for g, i in ck['stoi'].items():
        if g in stoi: proj[stoi[g]] = i
    NEG = -1e4; valid = proj >= 0
    vidx = torch.from_numpy(np.where(valid)[0]).to(dev); sidx = torch.from_numpy(proj[valid]).to(dev)

    def items(csvp):
        out = []
        for vid, g in read_csv(csvp):
            if g not in stoi: continue
            pp = os.path.join(args.pose_feat_dir, vid + '.pt'); sp = os.path.join(args.signrep_feat_dir, vid + '.npy')
            if os.path.exists(pp) and os.path.exists(sp): out.append((vid, stoi[g], pp, sp))
        return out
    data = {'val': items(args.val_csv), 'test': items(args.test_csv)}
    print(f"val={len(data['val'])} test={len(data['test'])}", flush=True)

    # cache clean SignRep canon log-probs + argmax per clip
    sr_cache = {}
    for split in ['val', 'test']:
        for vid, y, pp, sp in data[split]:
            sf = np.load(sp).astype(np.float32)
            if sf.shape[0] > args.max_seg: sf = sf[(sf.shape[0] - args.max_seg) // 2:][:args.max_seg]
            sx = torch.from_numpy(sf).unsqueeze(0).to(dev); sm = torch.ones(1, sf.shape[0], dtype=torch.bool, device=dev)
            sl = F.log_softmax(head(sx, sm), -1)[0]
            canon = torch.full((len(itos),), NEG, device=dev); canon[vidx] = sl[sidx]
            sr_cache[sp] = canon

    def run(split, ctype, sev, seed):
        rng = np.random.default_rng(seed); po, sr, fu, rel = [], [], [], []
        for vid, y, pp, sp in data[split]:
            pf = torch.load(pp, map_location='cpu').float().numpy()
            pc = corrupt_pose(pf, ctype, sev, rng)
            x = torch.from_numpy(pc).float().unsqueeze(0).to(dev)
            lp = F.log_softmax(pose(x, torch.tensor([pc.shape[0]], device=dev), labels=None)['logits'][0], -1)
            canon = sr_cache[sp]; fused = (1 - args.alpha) * lp + args.alpha * canon
            po.append(int(lp.argmax().item() == y)); sr.append(int(canon.argmax().item() == y))
            fu.append(int(fused.argmax().item() == y)); rel.append(float(pc[:, 258:261].mean()))
        return np.array(po), np.array(sr), np.array(fu), np.array(rel)

    cache = {}
    for split in ['val', 'test']:
        for ct in args.ctypes:
            for seed in args.seeds:
                for sev in args.sevs:
                    cache[(split, ct, sev, seed)] = run(split, ct, sev, seed)
                    print(f'ran {split} {ct} sev={sev} seed={seed}', flush=True)

    def gate_acc(key, tau):
        po, sr, fu, rel = cache[key]
        return float(np.where(rel < tau, sr, fu).mean())

    # tune tau on val
    allrel = np.concatenate([cache[k][3] for k in cache if k[0] == 'val'])
    best_tau, best = None, -1
    for tau in np.quantile(allrel, np.linspace(0, 1, 41)):
        a = np.mean([gate_acc(('val', ct, sev, seed), tau) for ct in args.ctypes for sev in args.sevs for seed in args.seeds])
        if a > best: best, best_tau = a, float(tau)
    print(f'selected tau={best_tau:.4f} (val gate acc {100*best:.2f})', flush=True)

    out = {'tau': best_tau, 'alpha': args.alpha}
    for ct in args.ctypes:
        rows = []
        for sev in args.sevs:
            cell = {'severity': sev}
            for nm, idx in [('pose', 0), ('signrep', 1), ('fusion', 2)]:
                v = [100 * cache[('test', ct, sev, seed)][idx].mean() for seed in args.seeds]
                cell[nm] = round(float(np.mean(v)), 2); cell[nm + '_std'] = round(float(np.std(v)), 2)
            v = [100 * gate_acc(('test', ct, sev, seed), best_tau) for seed in args.seeds]
            cell['gate'] = round(float(np.mean(v)), 2); cell['gate_std'] = round(float(np.std(v)), 2)
            rows.append(cell)
        out[ct] = rows
        print(f'[{ct}] ' + ' | '.join(f"s{r['severity']}: gate {r['gate']}±{r['gate_std']} fus {r['fusion']} sr {r['signrep']}" for r in rows), flush=True)
    json.dump(out, open(os.path.join(args.out_dir, 'final_islr.json'), 'w'), indent=2)
    print('RESULT:\n' + json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
