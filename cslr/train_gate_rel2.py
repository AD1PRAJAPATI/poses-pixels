import argparse, os, sys, json
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())
from modelling.model import build_model
from utils.misc import load_config, neq_load_customized, make_logger
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from utils.metrics import wer_list
from train_gate import load_pair, fuse

HANDS = list(range(91, 133))
CLEAN_A, FAIL_A = 0.45, 0.05


def pool_to_T(x, T):
    F = len(x)
    if F == T: return x
    idx = np.linspace(0, F, T + 1).astype(int)
    return np.array([x[idx[t]:max(idx[t] + 1, idx[t + 1])].mean() for t in range(T)])


def kp_cands(name):
    p = name.split('/'); c = [name]
    if len(p) >= 3: c.append(p[2])
    c.append(p[-1]); return c


def raw_conf(KP, name, T):
    kp = None
    for c in kp_cands(name):
        if c in KP: kp = np.array(KP[c]['keypoints']); break
    if kp is None: return np.full(T, 1.0)          # missing -> assume reliable
    conf = kp[:, HANDS, 2].mean(1)                  # (F,)
    return pool_to_T(conf, T).astype(np.float32)


def feats(lv, lk, rc):
    pv = lv.exp(); pk = lk.exp()
    vc = pv.max(-1).values; kc = pk.max(-1).values
    ve = -(pv * lv).sum(-1); ke = -(pk * lk).sum(-1)
    return torch.stack([rc, kc, ke, vc, ve], -1)    # (T,5)


class Gate(nn.Module):
    def __init__(self, d_in=5, h=32, alpha0=CLEAN_A):
        super().__init__()
        self.b0 = float(np.log(alpha0 / (1 - alpha0)))
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        return torch.sigmoid(self.b0 + self.net(f).squeeze(-1))


def attach_rawconf(items, KP):
    for it in items:
        it['rc'] = torch.from_numpy(raw_conf(KP, it['name'], it['lv'].shape[0]))


@torch.no_grad()
def eval_clean(items, decoder, beam, clean, device, gate=None):
    lower = decoder.gloss_tokenizer.lower_case
    hyps, refs = [], []
    for it in items:
        lv, lk, rc = it['lv'].to(device), it['lk'].to(device), it['rc'].to(device)
        a = gate(feats(lv, lk, rc)) if gate is not None else CLEAN_A
        fused = fuse(lv, lk, a).unsqueeze(0)
        ids = decoder.predict_gloss_from_logits(gloss_logits=fused, beam_size=beam,
              input_lengths=torch.tensor([fused.shape[1]], device=device))
        toks = decoder.gloss_tokenizer.convert_ids_to_tokens(ids)[0]
        hyp = ' '.join(toks).upper() if lower else ' '.join(toks)
        ref = it['gls_ref'].upper() if lower else it['gls_ref']
        hyps.append(clean(hyp)); refs.append(clean(ref))
    return wer_list(hypotheses=hyps, references=refs)['wer']


def train_epoch(gate, items, opt, device, bs, rng):
    gate.train(); np.random.shuffle(items); mse = nn.MSELoss(); tot = 0.0; n = 0
    for i in range(0, len(items), bs):
        preds, tgts = [], []
        for it in items[i:i + bs]:
            lv, lk = it['lv'].to(device), it['lk'].to(device)
            rc = it['rc'].clone()
            T = rc.shape[0]; frac = rng.uniform(0, 0.8); m = torch.from_numpy(rng.random(T) < frac)
            rc = rc.clone(); rc[m] = 0.0
            f = feats(lv, lk, rc.to(device))
            a = gate(f)
            tgt = torch.full((T,), CLEAN_A, device=device); tgt[m.to(device)] = FAIL_A
            preds.append(a); tgts.append(tgt)
        p = torch.cat(preds); t = torch.cat(tgts)
        loss = mse(p, t); opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * p.numel(); n += p.numel()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emis_dir', default='results/emissions')
    ap.add_argument('--keypoint_file', required=True)
    ap.add_argument('--decoder_config', required=True)
    ap.add_argument('--ckpt_name', default='best.ckpt')
    ap.add_argument('--out_dir', default='results/gate_rel2')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--beam', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='gate_rel2.log')
    device = torch.device('cuda'); rng = np.random.default_rng(0)

    cfg = load_config(args.decoder_config); cfg['device'] = device
    decoder = build_model(cfg)
    sd = torch.load(os.path.join(cfg['training']['model_dir'], 'ckpts', args.ckpt_name), map_location='cuda')
    neq_load_customized(decoder, sd['model_state'], verbose=False); decoder.eval()
    clean = clean_phoenix_2014_trans if cfg['data']['dataset_name'].lower() == 'phoenix-2014t' else clean_phoenix_2014

    import pickle
    KP = pickle.load(open(args.keypoint_file, 'rb'))
    train_items = load_pair(args.emis_dir, 'train'); attach_rawconf(train_items, KP)
    dev_items = load_pair(args.emis_dir, 'dev'); attach_rawconf(dev_items, KP)
    print(f'train={len(train_items)} dev={len(dev_items)}', flush=True)

    gate = Gate(alpha0=CLEAN_A).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f'[fixed 0.45] clean dev WER={eval_clean(dev_items, decoder, args.beam, clean, device):.2f}', flush=True)
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(gate, train_items, opt, device, args.bs, rng); sched.step()
        if ep % 10 == 0 or ep == args.epochs:
            print(f'ep {ep:3d} | mse {loss:.4f} | CLEAN dev WER {eval_clean(dev_items, decoder, args.beam, clean, device, gate):.2f}', flush=True)
    torch.save({k: v.cpu() for k, v in gate.state_dict().items()}, os.path.join(args.out_dir, 'gate.pt'))
    # probe: zero raw conf on random frames -> does alpha drop there?
    ac, ad = [], []
    for it in dev_items[:300]:
        lv, lk = it['lv'].to(device), it['lk'].to(device)
        rc = it['rc'].clone(); T = rc.shape[0]; m = torch.from_numpy(rng.random(T) < 0.5)
        with torch.no_grad():
            a0 = gate(feats(lv, lk, rc.to(device)))
            rc2 = rc.clone(); rc2[m] = 0.0
            a1 = gate(feats(lv, lk, rc2.to(device)))
        ac.append(a0.mean().item())
        if m.any(): ad.append(a1[m.to(device)].mean().item())
    print(f'\nsaved gate. probe: mean a (raw conf intact)={np.mean(ac):.3f}  '
          f'(raw conf ZEROED)={np.mean(ad):.3f}  (want ~0.45 vs ~0.05)', flush=True)
    json.dump({'ok': True}, open(os.path.join(args.out_dir, 'train_info.json'), 'w'))


if __name__ == '__main__':
    main()
