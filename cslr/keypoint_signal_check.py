import argparse, os, sys, pickle
import numpy as np
sys.path.append(os.getcwd())
BLANK = 0
BODY = list(range(0, 17)); FACE = list(range(23, 91))
LHAND = list(range(91, 112)); RHAND = list(range(112, 133)); HANDS = LHAND + RHAND


def align(lp, tgt, blank=BLANK):
    T = lp.shape[0]; ext = [blank]
    for y in tgt: ext += [int(y), blank]
    L = len(ext); NEG = -1e30
    dp = np.full((T, L), NEG); bp = np.zeros((T, L), int)
    dp[0, 0] = lp[0, ext[0]]
    if L > 1: dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(L):
            b, a = dp[t-1, s], s
            if s-1 >= 0 and dp[t-1, s-1] > b: b, a = dp[t-1, s-1], s-1
            if s-2 >= 0 and ext[s] != blank and ext[s] != ext[s-2] and dp[t-1, s-2] > b:
                b, a = dp[t-1, s-2], s-2
            dp[t, s] = b + lp[t, ext[s]]; bp[t, s] = a
    end = L-1 if dp[T-1, L-1] >= dp[T-1, max(0, L-2)] else L-2
    sp = np.zeros(T, int); si = end
    for t in range(T-1, -1, -1): sp[t] = ext[si]; si = bp[t, si]
    return sp


def load_emis(emis_dir, split):
    V = pickle.load(open(os.path.join(emis_dir, f'emis_video_{split}.pkl'), 'rb'))
    K = pickle.load(open(os.path.join(emis_dir, f'emis_keypoint_{split}.pkl'), 'rb'))
    return V, K, [n for n in V if n in K and V[n]['gloss_labels']]


def kp_candidates(name):
    parts = name.split('/'); c = [name]
    if len(parts) >= 3: c.append(parts[2])
    c.append(parts[-1]); return c


def frame_features(kp):
    F = kp.shape[0]
    xy = kp[:, :, :2].astype(np.float64)
    sc = kp[:, :, 2].astype(np.float64) if kp.shape[2] >= 3 else np.ones((F, kp.shape[1]))
    def conf(ids, agg): return agg(sc[:, ids], axis=1)
    hcm = conf(HANDS, np.mean); hci = conf(HANDS, np.min)
    bc = conf(BODY, np.mean); fca = conf(FACE, np.mean)
    hv = np.zeros(F); ha = np.zeros(F)
    d1 = np.linalg.norm(np.diff(xy[:, HANDS], axis=0), axis=2).mean(1); hv[1:] = d1
    if F > 2: ha[2:] = np.abs(np.diff(d1))
    lh = xy[:, LHAND].mean(1); rh = xy[:, RHAND].mean(1)
    inter = np.linalg.norm(lh - rh, axis=1)
    fc = xy[:, FACE].mean(1)
    hf = 0.5 * (np.linalg.norm(lh - fc, axis=1) + np.linalg.norm(rh - fc, axis=1))
    drop = (sc[:, HANDS] < 0.2).mean(1)
    return np.stack([hcm, hci, bc, fca, hv, ha, inter, hf, drop], 1)


def pool_to_T(feat, T):
    F = feat.shape[0]
    if F == T: return feat
    idx = np.linspace(0, F, T + 1).astype(int); out = np.zeros((T, feat.shape[1]))
    for t in range(T):
        lo, hi = idx[t], max(idx[t] + 1, idx[t + 1]); out[t] = feat[lo:min(hi, F)].mean(0)
    return out


def build_segments(V, K, names, KP, ms):
    Xs, ys = [], []
    for n in names:
        kp = None
        for c in kp_candidates(n):
            if c in KP: kp = np.array(KP[c]['keypoints'] if isinstance(KP[c],dict) else KP[c]); break
        if kp is None or kp.ndim != 3: ms['miss'] += 1; continue
        ms['hit'] += 1
        lv = V[n]['logp'][:].astype(np.float64); lk = K[n]['logp'][:].astype(np.float64)
        T = min(lv.shape[0], lk.shape[0]); lv, lk = lv[:T], lk[:T]
        s = align(0.55 * lv + 0.45 * lk, V[n]['gloss_labels'])
        idx = np.arange(T); pv = np.exp(lv[idx, s]); pk = np.exp(lk[idx, s])
        a = np.clip(pk / (pv + pk + 1e-8), 0.02, 0.98)
        ff = pool_to_T(frame_features(kp), T)
        t = 0
        while t < T:
            if s[t] != 0:
                j = t
                while j < T and s[j] == s[t]: j += 1
                Xs.append(ff[t:j].mean(0)); ys.append(a[t:j].mean()); t = j
            else: t += 1
    return np.array(Xs), np.array(ys)


def heldout_r2(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    beta, _, _, _ = np.linalg.lstsq(Xtr, ytr, rcond=None); pred = Xte @ beta
    return 1 - ((yte - pred) ** 2).sum() / (((yte - yte.mean()) ** 2).sum() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keypoint_file', required=True)
    ap.add_argument('--emis_dir', default='results/emissions')
    ap.add_argument('--fit_split', default='train')
    ap.add_argument('--eval_split', default='dev')
    args = ap.parse_args()
    print('loading keypoints...', flush=True)
    KP = pickle.load(open(args.keypoint_file, 'rb'))
    k0 = next(iter(KP)); v0 = np.array(KP[k0])
    print(f'keypoint keys={len(KP)} | sample key={repr(k0)[:80]} | shape={v0.shape} dtype={v0.dtype}', flush=True)
    ms = {'hit': 0, 'miss': 0}
    Vf, Kf, nf = load_emis(args.emis_dir, args.fit_split); Xtr, ytr = build_segments(Vf, Kf, nf, KP, ms)
    Ve, Ke, ne = load_emis(args.emis_dir, args.eval_split); Xte, yte = build_segments(Ve, Ke, ne, KP, ms)
    print(f'name-match: hit={ms["hit"]} miss={ms["miss"]}', flush=True)
    print(f'fit segments={len(Xtr)}  eval segments={len(Xte)}', flush=True)
    if len(Xtr) == 0 or len(Xte) == 0:
        print('NO SEGMENTS MATCHED -- keypoint key naming mismatch'); return
    nm = ['hand_conf_mean', 'hand_conf_min', 'body_conf', 'face_conf', 'hand_vel',
          'hand_acc', 'inter_hand', 'hand_face_dist', 'hand_dropout']
    print('\nheld-out per-feature corr (eval segments):')
    for i, x in enumerate(nm): print(f'  {x:16s}: {np.corrcoef(Xte[:, i], yte)[0,1]:+.3f}')
    r2 = heldout_r2(Xtr, ytr, Xte, yte)
    print(f'\nHELD-OUT R^2 (raw-pose features -> segment alpha): {r2:.3f}', flush=True)
    print('(posteriors gave ~0.01; >>0.01 means the signal is in the kinematics)')


if __name__ == '__main__':
    main()
