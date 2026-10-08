import argparse, os, sys, json, csv, pickle, copy
import numpy as np
import torch

sys.path.append(os.getcwd())
from dataset.Dataloader import build_dataloader
from utils.misc import move_to_device, make_logger
from utils.metrics import wer_list
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from train_gate import fuse
from train_gate_rel2 import HANDS
from degrade_eval import load_model, pick, KP_PRIORITY, S2G_PRIORITY

CLEAN = clean_phoenix_2014


def corrupt_split(KP, split, ctype, sev, seed):
    rng = np.random.default_rng(seed); out = {}
    for n, v in KP.items():
        if f'/{split}/' not in n: continue
        kp = np.array(v['keypoints']).astype(np.float32).copy(); F = kp.shape[0]
        if sev > 0:
            if ctype == 'dropout':
                m = rng.random(F) < sev; kp[m] = 0.0
            elif ctype == 'hand_occl':
                m = rng.random(F) < sev
                kp[np.ix_(np.where(m)[0], list(range(91, 133)))] = 0.0
            elif ctype == 'noise':
                s = sev * (kp[:, :, :2].std() + 1e-6)
                kp[:, :, :2] += rng.normal(0, s, kp[:, :, :2].shape).astype(np.float32)
        out[n] = {'keypoints': kp}
    return out


@torch.no_grad()
def run_stream(model, cfg, kp_file, split, priority, device):
    cfg = copy.deepcopy(cfg); cfg['data']['keypoint_file'] = kp_file; cfg['training']['batch_size'] = 1
    dl, _ = build_dataloader(cfg, split, model.text_tokenizer, model.gloss_tokenizer)
    out = {}
    for batch in dl:
        batch = move_to_device(batch, device)
        fo = model(is_train=False, **batch)
        k = pick(fo, priority)
        out[batch['name'][0]] = dict(logp=torch.log_softmax(fo[k], -1)[0].float().cpu().numpy(),
                                     length=int(fo['input_lengths'][0].item()), gls_ref=batch['gloss'][0])
    return out


def decode(logp, decoder, beam, device):
    lower = decoder.gloss_tokenizer.lower_case
    ids = decoder.predict_gloss_from_logits(gloss_logits=torch.from_numpy(logp).to(device).unsqueeze(0),
          beam_size=beam, input_lengths=torch.tensor([logp.shape[0]], device=device))
    toks = decoder.gloss_tokenizer.convert_ids_to_tokens(ids)[0]
    return CLEAN((' '.join(toks).upper()) if lower else ' '.join(toks))


def wer(hyps, refs): return wer_list(hypotheses=hyps, references=refs)['wer']


def collect(kp_model, kp_cfg, s2g_model, s2g_cfg, KP, video_emis, split, ctype, sev, seed, alpha, beam, device, tmp):
    """Return per-clip dict: name -> {video,keypoint,fixed,cotrain hyps; ref; conf}."""
    corr = corrupt_split(KP, split, ctype, sev, seed)
    pickle.dump(corr, open(tmp, 'wb'))
    kp_emis = run_stream(kp_model, kp_cfg, tmp, split, KP_PRIORITY, device)
    s2g_emis = run_stream(s2g_model, s2g_cfg, tmp, split, S2G_PRIORITY, device)
    os.remove(tmp)
    lower = kp_model.gloss_tokenizer.lower_case
    rows = {}
    for n, d in kp_emis.items():
        if n not in video_emis or n not in s2g_emis: continue
        lk = d['logp'].astype(np.float32); lv = video_emis[n]['logp'].astype(np.float32)
        T = min(lk.shape[0], lv.shape[0])
        fused = (1 - alpha) * lv[:T] + alpha * lk[:T]   # emissions are already log-probs
        ref = CLEAN((d['gls_ref'].upper()) if lower else d['gls_ref'])
        kp_arr = np.array(corr[n]['keypoints'])
        rows[n] = dict(
            video=decode(lv, kp_model, beam, device),
            keypoint=decode(lk, kp_model, beam, device),
            fixed=decode(fused, kp_model, beam, device),
            cotrain=decode(s2g_emis[n]['logp'].astype(np.float32), kp_model, beam, device),
            ref=ref, conf=float(kp_arr[:, HANDS, 2].mean()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keypoint_config', required=True); ap.add_argument('--s2g_config', required=True)
    ap.add_argument('--emis_dir', default='results/emissions'); ap.add_argument('--out_dir', default='results/final')
    ap.add_argument('--ctypes', nargs='+', default=['dropout', 'hand_occl'])
    ap.add_argument('--sevs', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--alpha', type=float, default=0.45); ap.add_argument('--beam', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='final.log')
    device = torch.device('cuda')

    kp_model, kp_cfg = load_model(args.keypoint_config)
    s2g_model, s2g_cfg = load_model(args.s2g_config)
    KP = pickle.load(open(kp_cfg['data']['keypoint_file'], 'rb'))
    v_dev = pickle.load(open(os.path.join(args.emis_dir, 'emis_video_dev.pkl'), 'rb'))
    v_test = pickle.load(open(os.path.join(args.emis_dir, 'emis_video_test.pkl'), 'rb'))
    tmp = os.path.join(args.out_dir, '_tmp.pkl')

    # cache all per-clip rows (saved to disk so a rerun skips the expensive collection)
    cache_path = os.path.join(args.out_dir, 'cache.pkl')
    if os.path.exists(cache_path):
        cache = pickle.load(open(cache_path, 'rb')); print('loaded cached collection', flush=True)
    else:
        cache = {}  # (split,ctype,sev,seed) -> rows
        for ct in args.ctypes:
            for seed in args.seeds:
                for split, vem in [('dev', v_dev), ('test', v_test)]:
                    for sev in args.sevs:
                        cache[(split, ct, sev, seed)] = collect(kp_model, kp_cfg, s2g_model, s2g_cfg, KP, vem,
                                                                split, ct, sev, seed, args.alpha, args.beam, device, tmp)
                        print(f'collected {split} {ct} sev={sev} seed={seed}', flush=True)
        pickle.dump(cache, open(cache_path, 'wb'))

    def method_wer(rows, which, tau=None):
        if which == 'gate':
            hyp = [r['video'] if r['conf'] < tau else r['cotrain'] for r in rows.values()]
        else:
            hyp = [r[which] for r in rows.values()]
        ref = [r['ref'] for r in rows.values()]
        return wer(hyp, ref)

    # tune a single tau on dev (minimise mean dev WER over ctypes/sevs/seeds)
    all_conf = np.concatenate([[r['conf'] for r in cache[k].values()] for k in cache if k[0] == 'dev'])
    best_tau, best = None, 1e9
    for tau in np.quantile(all_conf, np.linspace(0, 1, 41)):
        ws = [method_wer(cache[('dev', ct, sev, seed)], 'gate', tau)
              for ct in args.ctypes for sev in args.sevs for seed in args.seeds]
        if np.mean(ws) < best: best, best_tau = np.mean(ws), float(tau)
    print(f'selected tau={best_tau:.4f} (dev mean WER {best:.2f})', flush=True)

    # report on TEST, mean +/- std over seeds
    out = {'tau': best_tau, 'alpha': args.alpha}
    for ct in args.ctypes:
        rows_out = []
        for sev in args.sevs:
            cell = {'severity': sev}
            for meth in ['video', 'keypoint', 'fixed', 'cotrain']:
                vals = [method_wer(cache[('test', ct, sev, seed)], meth) for seed in args.seeds]
                cell[meth] = round(float(np.mean(vals)), 2); cell[meth + '_std'] = round(float(np.std(vals)), 2)
            vals = [method_wer(cache[('test', ct, sev, seed)], 'gate', best_tau) for seed in args.seeds]
            cell['gate'] = round(float(np.mean(vals)), 2); cell['gate_std'] = round(float(np.std(vals)), 2)
            rows_out.append(cell)
        out[ct] = rows_out
    json.dump(out, open(os.path.join(args.out_dir, 'final_cslr.json'), 'w'), indent=2)
    for ct in args.ctypes:
        for r in out[ct]:
            print(f"[{ct} s{r['severity']}] gate {r['gate']}+-{r['gate_std']} cot {r['cotrain']}+-{r['cotrain_std']} fix {r['fixed']} vid {r['video']}", flush=True)
    print('RESULT:\n' + json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
