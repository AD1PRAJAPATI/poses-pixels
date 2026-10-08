import argparse, os, sys, pickle, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
from modelling.model import build_model
from utils.misc import load_config, neq_load_customized, make_logger
from utils.metrics import wer_list
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans

BLANK = 0


def load_pair(emis_dir, split):
    v = pickle.load(open(os.path.join(emis_dir, f'emis_video_{split}.pkl'), 'rb'))
    k = pickle.load(open(os.path.join(emis_dir, f'emis_keypoint_{split}.pkl'), 'rb'))
    names = [n for n in v if n in k and v[n]['gloss_labels']]
    items = []
    for n in names:
        T = min(v[n]['logp'].shape[0], k[n]['logp'].shape[0])
        items.append(dict(
            name=n,
            lv=torch.from_numpy(v[n]['logp'][:T].astype(np.float32)),
            lk=torch.from_numpy(k[n]['logp'][:T].astype(np.float32)),
            length=T,
            target=torch.tensor(v[n]['gloss_labels'], dtype=torch.long),
            gls_ref=v[n]['gls_ref']))
    return items


def frame_feats(lp):
    """lp: (T,V) log-probs -> (T,3): entropy, top-1 margin, top-1 confidence."""
    p = lp.exp()
    conf, idx = p.max(-1)
    p2 = p.clone().scatter_(1, idx.unsqueeze(1), 0.0)
    margin = conf - p2.max(-1).values
    ent = -(p * lp).sum(-1)
    return torch.stack([ent, margin, conf], -1)


class Gate(nn.Module):
    """Residual gate: alpha_t = sigmoid(logit(alpha0) + delta_t).
    Output layer zero-initialized so the gate STARTS exactly at the fixed
    baseline (alpha_t == alpha0 for all t) and can only learn to improve."""
    def __init__(self, d_in=6, h=32, alpha0=0.45):
        super().__init__()
        self.b0 = math.log(alpha0 / (1 - alpha0))
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, fv, fk):                       # (T,3),(T,3) -> (T,)
        delta = self.net(torch.cat([fv, fk], -1)).squeeze(-1)
        return torch.sigmoid(self.b0 + delta)


def fuse(lv, lk, alpha):                             # alpha: scalar or (T,)
    if not torch.is_tensor(alpha):
        alpha = torch.full((lv.shape[0],), float(alpha), device=lv.device)
    a = alpha.unsqueeze(-1)
    return F.log_softmax((1 - a) * lv + a * lk, dim=-1)   # (T,V)


def train_epoch(gate, items, opt, device, bs, smooth_lambda):
    gate.train()
    np.random.shuffle(items)
    tot = 0.0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        fused_list, tgts, in_len, tg_len, smooth = [], [], [], [], []
        for it in chunk:
            lv, lk = it['lv'].to(device), it['lk'].to(device)
            a = gate(frame_feats(lv), frame_feats(lk))       # (T,)
            fused_list.append(fuse(lv, lk, a))               # (T,V)
            if a.shape[0] > 1:
                smooth.append((a[1:] - a[:-1]).abs().mean())  # temporal smoothness
            tgts.append(it['target'].to(device))
            in_len.append(it['length']); tg_len.append(len(it['target']))
        Tmax = max(f.shape[0] for f in fused_list); V = fused_list[0].shape[1]
        padded = torch.full((len(chunk), Tmax, V), 0.0, device=device)
        for j, f in enumerate(fused_list):
            padded[j, :f.shape[0]] = f
        logp = padded.permute(1, 0, 2)                       # (T,B,V)
        loss = F.ctc_loss(logp, torch.cat(tgts),
                          torch.tensor(in_len, device=device),
                          torch.tensor(tg_len, device=device),
                          blank=BLANK, zero_infinity=True)
        if smooth_lambda > 0 and smooth:
            loss = loss + smooth_lambda * torch.stack(smooth).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(chunk)
    return tot / len(items)


@torch.no_grad()
def evaluate(items, decoder, beam, clean, device, gate=None, fixed_alpha=None):
    lower = decoder.gloss_tokenizer.lower_case
    hyps, refs, all_alpha = [], [], []
    for it in items:
        lv, lk = it['lv'].to(device), it['lk'].to(device)
        if gate is not None:
            a = gate(frame_feats(lv), frame_feats(lk))
            all_alpha.append(a.detach().cpu().numpy())
        else:
            a = fixed_alpha
        fused = fuse(lv, lk, a).unsqueeze(0)                 # (1,T,V)
        il = torch.tensor([it['length']], device=device)
        ids = decoder.predict_gloss_from_logits(gloss_logits=fused, beam_size=beam, input_lengths=il)
        toks = decoder.gloss_tokenizer.convert_ids_to_tokens(ids)[0]
        hyp = ' '.join(toks).upper() if lower else ' '.join(toks)
        ref = it['gls_ref'].upper() if lower else it['gls_ref']
        hyps.append(clean(hyp)); refs.append(clean(ref))
    wer = wer_list(hypotheses=hyps, references=refs)['wer']
    dist = None
    if all_alpha:
        av = np.concatenate(all_alpha)
        pct = np.percentile(av, [0, 10, 25, 50, 75, 90, 100])
        frac_extreme = float(((av < 0.05) | (av > 0.95)).mean())   # sigmoid-saturation check
        dist = {'mean': float(av.mean()), 'std': float(av.std()),
                'pctl_0_10_25_50_75_90_100': [round(float(x), 3) for x in pct],
                'frac_saturated': round(frac_extreme, 3)}
    return wer, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emis_dir', default='results/emissions')
    ap.add_argument('--decoder_config', required=True)
    ap.add_argument('--ckpt_name', default='best.ckpt')
    ap.add_argument('--out_dir', default='results/gate')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--fixed_alpha', type=float, default=0.45)
    ap.add_argument('--alpha0', type=float, default=0.45, help='residual init: gate starts here')
    ap.add_argument('--smooth_lambda', type=float, default=0.1, help='penalty on |a_t - a_{t-1}|')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='gate.log')   # SLRT get_logger() needs this before build_model
    device = torch.device('cuda')

    # decoder model (frozen) -> gloss_tokenizer + CTC beam decode, same path as sweep
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

    # fixed-alpha reference (should reproduce the sweep's ~29.45 at alpha=0.45)
    dev_fx, _ = evaluate(dev_items, decoder, args.beam, clean, device, fixed_alpha=args.fixed_alpha)
    print(f'[fixed a={args.fixed_alpha}] dev WER={dev_fx:.2f}', flush=True)

    gate = Gate(alpha0=args.alpha0).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    # sanity: at init the gate should reproduce the fixed baseline exactly
    dev0, dist0 = evaluate(dev_items, decoder, args.beam, clean, device, gate=gate)
    print(f'[gate init] dev WER={dev0:.2f} (should ~= fixed {dev_fx:.2f}) | a-dist {dist0}', flush=True)
    best_dev, best_state = 1e9, None
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(gate, train_items, opt, device, args.bs, args.smooth_lambda)
        sched.step()
        dev_wer, dev_a = evaluate(dev_items, decoder, args.beam, clean, device, gate=gate)
        print(f'ep {ep:3d} | loss {loss:.3f} | dev WER {dev_wer:.2f} | a-dist {dev_a}', flush=True)
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
