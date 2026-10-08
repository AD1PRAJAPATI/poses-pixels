import argparse, os, sys, json
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())
from modelling.model import build_model
from utils.misc import load_config, neq_load_customized, make_logger
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from train_gate import load_pair, frame_feats, Gate, fuse, evaluate

BLANK = 0


def ctc_forced_align(lp, target, blank=BLANK):
    """lp: (T,V) log-probs (any per-frame constant offset is fine).
    target: list of gloss ids (no blanks). Returns s_t: (T,) aligned symbol ids."""
    T = lp.shape[0]
    ext = [blank]
    for y in target:
        ext += [int(y), blank]
    L = len(ext)
    NEG = -1e30
    dp = np.full((T, L), NEG, dtype=np.float64)
    bp = np.zeros((T, L), dtype=np.int32)
    dp[0, 0] = lp[0, ext[0]]
    if L > 1:
        dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(L):
            best, arg = dp[t - 1, s], s
            if s - 1 >= 0 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], s - 1
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best:
                best, arg = dp[t - 1, s - 2], s - 2
            dp[t, s] = best + lp[t, ext[s]]
            bp[t, s] = arg
    end = L - 1 if dp[T - 1, L - 1] >= dp[T - 1, max(0, L - 2)] else L - 2
    s_path = np.zeros(T, dtype=np.int64)
    si = end
    for t in range(T - 1, -1, -1):
        s_path[t] = ext[si]
        si = bp[t, si]
    return s_path


def build_alpha_targets(items, fixed_alpha):
    """Attach it['alpha_target'] (T,) in [0.02,0.98] via forced alignment."""
    for it in items:
        lv = it['lv'].numpy().astype(np.float64)          # (T,V) log-probs
        lk = it['lk'].numpy().astype(np.float64)
        tgt = it['target'].tolist()
        if len(tgt) == 0:
            it['alpha_target'] = torch.full((lv.shape[0],), fixed_alpha); continue
        align_lp = (1 - fixed_alpha) * lv + fixed_alpha * lk
        s = ctc_forced_align(align_lp, tgt)               # (T,)
        idx = np.arange(lv.shape[0])
        pv = np.exp(lv[idx, s]); pk = np.exp(lk[idx, s])
        a = pk / (pv + pk + 1e-8)
        it['alpha_target'] = torch.tensor(np.clip(a, 0.02, 0.98), dtype=torch.float32)


def train_epoch(gate, items, opt, device, bs):
    gate.train(); np.random.shuffle(items)
    mse = nn.MSELoss(); tot = 0.0; n = 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        preds, tgts = [], []
        for it in chunk:
            lv, lk = it['lv'].to(device), it['lk'].to(device)
            a = gate(frame_feats(lv), frame_feats(lk))    # (T,)
            preds.append(a); tgts.append(it['alpha_target'].to(device))
        p = torch.cat(preds); t = torch.cat(tgts)
        loss = mse(p, t)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * p.numel(); n += p.numel()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emis_dir', default='results/emissions')
    ap.add_argument('--decoder_config', required=True)
    ap.add_argument('--ckpt_name', default='best.ckpt')
    ap.add_argument('--out_dir', default='results/gate_ds')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--fixed_alpha', type=float, default=0.45)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='gate_ds.log')
    device = torch.device('cuda')

    cfg = load_config(args.decoder_config); cfg['device'] = device
    decoder = build_model(cfg)
    sd = torch.load(os.path.join(cfg['training']['model_dir'], 'ckpts', args.ckpt_name), map_location='cuda')
    neq_load_customized(decoder, sd['model_state'], verbose=False); decoder.eval()
    dsname = cfg['data']['dataset_name'].lower()
    clean = clean_phoenix_2014_trans if dsname == 'phoenix-2014t' else clean_phoenix_2014

    train_items = load_pair(args.emis_dir, 'train')
    dev_items = load_pair(args.emis_dir, 'dev')
    test_items = load_pair(args.emis_dir, 'test')
    print(f'loaded train={len(train_items)} dev={len(dev_items)} test={len(test_items)}', flush=True)
    print('building alpha targets via forced alignment...', flush=True)
    build_alpha_targets(train_items, args.fixed_alpha)
    at = np.concatenate([it['alpha_target'].numpy() for it in train_items])
    print(f'alpha_target dist: mean={at.mean():.3f} std={at.std():.3f} '
          f'pctl10/50/90={np.percentile(at,[10,50,90]).round(3).tolist()}', flush=True)

    dev_fx, _ = evaluate(dev_items, decoder, args.beam, clean, device, fixed_alpha=args.fixed_alpha)
    print(f'[fixed a={args.fixed_alpha}] dev WER={dev_fx:.2f}', flush=True)

    gate = Gate(alpha0=args.fixed_alpha).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    dev0, dist0 = evaluate(dev_items, decoder, args.beam, clean, device, gate=gate)
    print(f'[gate init] dev WER={dev0:.2f} (should ~= {dev_fx:.2f})', flush=True)

    best_dev, best_state = 1e9, None
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(gate, train_items, opt, device, args.bs)
        sched.step()
        dev_wer, dev_a = evaluate(dev_items, decoder, args.beam, clean, device, gate=gate)
        print(f'ep {ep:3d} | mse {loss:.4f} | dev WER {dev_wer:.2f} | a-dist {dev_a}', flush=True)
        if dev_wer < best_dev:
            best_dev = dev_wer
            best_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}
            torch.save(best_state, os.path.join(args.out_dir, 'gate_best.pt'))

    gate.load_state_dict(best_state)
    test_wer, test_a = evaluate(test_items, decoder, args.beam, clean, device, gate=gate)
    test_fx, _ = evaluate(test_items, decoder, args.beam, clean, device, fixed_alpha=args.fixed_alpha)
    out = {'gate_dev_wer': best_dev, 'gate_test_wer': test_wer, 'gate_test_alpha_dist': test_a,
           'fixed_alpha': args.fixed_alpha, 'fixed_dev_wer': dev_fx, 'fixed_test_wer': test_fx}
    json.dump(out, open(os.path.join(args.out_dir, 'gate_result.json'), 'w'), indent=2)
    print('\nRESULT: ' + json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
