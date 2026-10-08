import argparse, os, sys, json, csv, pickle, tempfile
import numpy as np
import torch

sys.path.append(os.getcwd())
from modelling.model import build_model
from dataset.Dataloader import build_dataloader
from utils.misc import load_config, move_to_device, neq_load_customized, make_logger
from utils.metrics import wer_list
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans

KP_PRIORITY = ['gloss_logits', 'keypoint_gloss_logits']
S2G_PRIORITY = ['ensemble_last_gloss_logits', 'fuse_gloss_logits']


def pick(fo, pr):
    for k in pr:
        if k in fo and fo[k] is not None: return k
    for k in fo:
        if 'gloss_logits' in k and fo[k] is not None: return k
    raise KeyError(str(list(fo.keys())))


def load_model(cfg_path):
    cfg = load_config(cfg_path); cfg['device'] = torch.device('cuda')
    cfg['training']['batch_size'] = 1
    m = build_model(cfg)
    sd = torch.load(os.path.join(cfg['training']['model_dir'], 'ckpts', 'best.ckpt'), map_location='cuda')
    neq_load_customized(m, sd['model_state'], verbose=False); m.eval()
    return m, cfg


def corrupt_dev_keypoints(KP, ctype, sev, seed=0):
    """Return a dev-only dict {name: {'keypoints': corrupted (F,133,3)}}."""
    rng = np.random.default_rng(seed)
    out = {}
    for n, v in KP.items():
        if '/dev/' not in n: continue
        kp = np.array(v['keypoints']).astype(np.float32).copy()
        F = kp.shape[0]
        if sev > 0:
            if ctype == 'dropout':                       # whole-frame detection failure
                m = rng.random(F) < sev
                kp[m, :, :2] = 0.0; kp[m, :, 2] = 0.0
            elif ctype == 'hand_occl':                   # hands undetected on some frames
                m = rng.random(F) < sev
                kp[np.ix_(np.where(m)[0], list(range(91, 133)))] = 0.0
            elif ctype == 'noise':                       # gaussian coord noise, sev * coord-std
                s = sev * (kp[:, :, :2].std() + 1e-6)
                kp[:, :, :2] += rng.normal(0, s, kp[:, :, :2].shape).astype(np.float32)
        out[n] = {'keypoints': kp}
    return out


@torch.no_grad()
def run_emis(model, cfg, kp_file, priority):
    cfg = dict(cfg); cfg['data'] = dict(cfg['data']); cfg['data']['keypoint_file'] = kp_file
    dl, _ = build_dataloader(cfg, 'dev', model.text_tokenizer, model.gloss_tokenizer)
    out = {}
    for batch in dl:
        batch = move_to_device(batch, cfg['device'])
        fo = model(is_train=False, **batch)
        k = pick(fo, priority)
        out[batch['name'][0]] = dict(
            logp=torch.log_softmax(fo[k], -1)[0].float().cpu().numpy(),
            length=int(fo['input_lengths'][0].item()),
            gls_ref=batch['gloss'][0])
    return out


def decode_wer(emis, decoder, beam, clean, device, fuse_with=None, alpha=0.45):
    lower = decoder.gloss_tokenizer.lower_case
    hyps, refs = [], []
    for n, d in emis.items():
        lp = torch.from_numpy(d['logp']).to(device)
        if fuse_with is not None and n in fuse_with:
            lv = torch.from_numpy(fuse_with[n]['logp']).to(device)
            T = min(lv.shape[0], lp.shape[0])
            lp = torch.log_softmax((1 - alpha) * lv[:T] + alpha * lp[:T], -1)
        ids = decoder.predict_gloss_from_logits(gloss_logits=lp.unsqueeze(0),
              beam_size=beam, input_lengths=torch.tensor([lp.shape[0]], device=device))
        toks = decoder.gloss_tokenizer.convert_ids_to_tokens(ids)[0]
        hyp = ' '.join(toks).upper() if lower else ' '.join(toks)
        ref = d['gls_ref'].upper() if lower else d['gls_ref']
        hyps.append(clean(hyp)); refs.append(clean(ref))
    return wer_list(hypotheses=hyps, references=refs)['wer']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keypoint_config', required=True)
    ap.add_argument('--s2g_config', required=True)
    ap.add_argument('--video_emis', required=True)
    ap.add_argument('--ctype', default='dropout', choices=['dropout', 'hand_occl', 'noise'])
    ap.add_argument('--sevs', type=float, nargs='+', default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument('--alpha', type=float, default=0.45)
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--out_dir', default='results/degrade')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='degrade.log')
    device = torch.device('cuda')

    video_emis = pickle.load(open(args.video_emis, 'rb'))
    kp_model, kp_cfg = load_model(args.keypoint_config)
    s2g_model, s2g_cfg = load_model(args.s2g_config)
    clean = clean_phoenix_2014_trans if kp_cfg['data']['dataset_name'].lower() == 'phoenix-2014t' else clean_phoenix_2014
    KP = pickle.load(open(kp_cfg['data']['keypoint_file'], 'rb'))

    video_only = decode_wer(video_emis, kp_model, args.beam, clean, device)  # clean, constant
    rows = []
    for sev in args.sevs:
        corr = corrupt_dev_keypoints(KP, args.ctype, sev)
        tf = os.path.join(args.out_dir, f'_kp_{args.ctype}_{sev}.pkl')
        pickle.dump(corr, open(tf, 'wb'))
        kp_emis = run_emis(kp_model, kp_cfg, tf, KP_PRIORITY)
        kp_only = decode_wer(kp_emis, kp_model, args.beam, clean, device)
        fixed = decode_wer(kp_emis, kp_model, args.beam, clean, device, fuse_with=video_emis, alpha=args.alpha)
        s2g_emis = run_emis(s2g_model, s2g_cfg, tf, S2G_PRIORITY)
        cotrain = decode_wer(s2g_emis, s2g_model, args.beam, clean, device)
        os.remove(tf)
        row = dict(severity=sev, video_only=video_only, keypoint_only=kp_only,
                   fixed_fusion=fixed, co_training=cotrain)
        rows.append(row)
        print(f'[{args.ctype} sev={sev}] video={video_only:.2f} kp={kp_only:.2f} '
              f'fixed={fixed:.2f} cotrain={cotrain:.2f}', flush=True)

    with open(os.path.join(args.out_dir, f'degrade_{args.ctype}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(os.path.join(args.out_dir, f'degrade_{args.ctype}.json'), 'w'), indent=2)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        xs = [r['severity'] for r in rows]
        plt.figure(figsize=(5, 3.4))
        for key, lab, st in [('video_only', 'video-only', 'k--'), ('keypoint_only', 'keypoint-only', 'C1-o'),
                             ('fixed_fusion', 'fixed fusion', 'C0-o'), ('co_training', 'co-training', 'C3-o')]:
            plt.plot(xs, [r[key] for r in rows], st, label=lab, ms=3)
        plt.xlabel(f'{args.ctype} severity'); plt.ylabel('dev WER'); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'degrade_{args.ctype}.png'), dpi=150)
        print('wrote figure', flush=True)
    except Exception as e:
        print('figure skipped:', e)
    print('RESULT:', json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
