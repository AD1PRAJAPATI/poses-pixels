import argparse, os, sys, json
import numpy as np
import torch

sys.path.append(os.getcwd())
from modelling.model import build_model
from utils.misc import load_config, neq_load_customized, make_logger
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from utils.metrics import wer_list
from train_gate import load_pair, fuse
from train_gate_ds import build_alpha_targets, ctc_forced_align


def moving_average(a, w):
    if w <= 1:
        return a
    pad = w // 2
    ap = np.pad(a, pad, mode='edge')
    ker = np.ones(w) / w
    return np.convolve(ap, ker, mode='valid')[:len(a)]


def per_sign_alpha(a, lv, lk, tgt, fixed_alpha):
    """Constant alpha within each aligned gloss segment (segment mean); blanks=fixed."""
    T = len(a)
    s = ctc_forced_align((1 - fixed_alpha) * lv + fixed_alpha * lk, tgt)
    out = np.full(T, fixed_alpha)
    t = 0
    while t < T:
        if s[t] != 0:
            j = t
            while j < T and s[j] == s[t]:
                j += 1
            out[t:j] = a[t:j].mean()
            t = j
        else:
            t += 1
    return out


@torch.no_grad()
def decode_wer(items, decoder, beam, clean, device, alpha_key):
    lower = decoder.gloss_tokenizer.lower_case
    hyps, refs = [], []
    for it in items:
        lv, lk = it['lv'].to(device), it['lk'].to(device)
        a = it[alpha_key]
        a = a.to(device) if torch.is_tensor(a) else torch.tensor(a, device=device, dtype=torch.float32)
        fused = fuse(lv, lk, a).unsqueeze(0)
        il = torch.tensor([it['length']], device=device)
        ids = decoder.predict_gloss_from_logits(gloss_logits=fused, beam_size=beam, input_lengths=il)
        toks = decoder.gloss_tokenizer.convert_ids_to_tokens(ids)[0]
        hyp = ' '.join(toks).upper() if lower else ' '.join(toks)
        ref = it['gls_ref'].upper() if lower else it['gls_ref']
        hyps.append(clean(hyp)); refs.append(clean(ref))
    return wer_list(hypotheses=hyps, references=refs)['wer']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emis_dir', default='results/emissions')
    ap.add_argument('--decoder_config', required=True)
    ap.add_argument('--ckpt_name', default='best.ckpt')
    ap.add_argument('--out_dir', default='results/gate_ds')
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--fixed_alpha', type=float, default=0.45)
    ap.add_argument('--windows', type=int, nargs='+', default=[3, 5, 9, 15])
    ap.add_argument('--splits', nargs='+', default=['dev', 'test'])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='oracle_variants.log')
    device = torch.device('cuda')

    cfg = load_config(args.decoder_config); cfg['device'] = device
    decoder = build_model(cfg)
    sd = torch.load(os.path.join(cfg['training']['model_dir'], 'ckpts', args.ckpt_name), map_location='cuda')
    neq_load_customized(decoder, sd['model_state'], verbose=False); decoder.eval()
    clean = clean_phoenix_2014_trans if cfg['data']['dataset_name'].lower() == 'phoenix-2014t' else clean_phoenix_2014

    result = {}
    for split in args.splits:
        items = load_pair(args.emis_dir, split)
        build_alpha_targets(items, args.fixed_alpha)   # sets it['alpha_target'] (raw oracle)
        # precompute variant alpha schedules per clip
        for it in items:
            a = it['alpha_target'].numpy()
            lv = it['lv'].numpy().astype(np.float64); lk = it['lk'].numpy().astype(np.float64)
            it['a_fixed'] = np.full(len(a), args.fixed_alpha, dtype=np.float32)
            it['a_oracle'] = a.astype(np.float32)
            for w in args.windows:
                it[f'a_smooth{w}'] = moving_average(a, w).astype(np.float32)
            it['a_persign'] = per_sign_alpha(a, lv, lk, it['target'].tolist(), args.fixed_alpha).astype(np.float32)

        r = {}
        r['fixed'] = decode_wer(items, decoder, args.beam, clean, device, 'a_fixed')
        r['oracle'] = decode_wer(items, decoder, args.beam, clean, device, 'a_oracle')
        r['per_sign'] = decode_wer(items, decoder, args.beam, clean, device, 'a_persign')
        for w in args.windows:
            r[f'smooth{w}'] = decode_wer(items, decoder, args.beam, clean, device, f'a_smooth{w}')
        result[split] = r
        print(f'[{split}] ' + '  '.join(f'{k}={v:.2f}' for k, v in r.items()), flush=True)

    json.dump(result, open(os.path.join(args.out_dir, 'oracle_variants.json'), 'w'), indent=2)
    print('RESULT: ' + json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
