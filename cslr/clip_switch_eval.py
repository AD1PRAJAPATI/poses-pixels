import argparse, os, sys, json, csv, pickle
import numpy as np
import torch

sys.path.append(os.getcwd())
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from utils.metrics import wer_list
from utils.misc import make_logger
from train_gate import fuse
from train_gate_rel2 import pool_to_T, HANDS
from degrade_eval import load_model, corrupt_dev_keypoints, run_emis, KP_PRIORITY


def decode_one(logp, decoder, beam, device, lower):
    ids = decoder.predict_gloss_from_logits(gloss_logits=logp.unsqueeze(0), beam_size=beam,
          input_lengths=torch.tensor([logp.shape[0]], device=device))
    toks = decoder.gloss_tokenizer.convert_ids_to_tokens(ids)[0]
    return ' '.join(toks).upper() if lower else ' '.join(toks)


def wer(hyps, refs):
    return wer_list(hypotheses=hyps, references=refs)['wer']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keypoint_config', required=True)
    ap.add_argument('--video_emis', required=True)
    ap.add_argument('--ctype', default='dropout')
    ap.add_argument('--sevs', type=float, nargs='+', default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument('--baseline_json', default=None)
    ap.add_argument('--alpha', type=float, default=0.45)
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--out_dir', default='results/clip_switch')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='clip_switch.log')
    device = torch.device('cuda')

    video_emis = pickle.load(open(args.video_emis, 'rb'))
    kp_model, kp_cfg = load_model(args.keypoint_config)
    lower = kp_model.gloss_tokenizer.lower_case
    clean = clean_phoenix_2014_trans if kp_cfg['data']['dataset_name'].lower() == 'phoenix-2014t' else clean_phoenix_2014
    KP = pickle.load(open(kp_cfg['data']['keypoint_file'], 'rb'))
    baseline = {r['severity']: r for r in json.load(open(args.baseline_json))} if args.baseline_json and os.path.exists(args.baseline_json) else {}

    rows = []
    for sev in args.sevs:
        corr = corrupt_dev_keypoints(KP, args.ctype, sev)
        tf = os.path.join(args.out_dir, f'_kp_{sev}.pkl'); pickle.dump(corr, open(tf, 'wb'))
        kp_emis = run_emis(kp_model, kp_cfg, tf, KP_PRIORITY); os.remove(tf)
        names, v_hyp, f_hyp, refs, conf = [], [], [], [], []
        for n, d in kp_emis.items():
            if n not in video_emis: continue
            lk = torch.from_numpy(d['logp']).to(device); lv = torch.from_numpy(video_emis[n]['logp']).to(device)
            T = min(lv.shape[0], lk.shape[0]); lv, lk = lv[:T], lk[:T]
            v_hyp.append(clean(decode_one(torch.log_softmax(lv, -1), kp_model, args.beam, device, lower)))
            f_hyp.append(clean(decode_one(fuse(lv, lk, args.alpha), kp_model, args.beam, device, lower)))
            refs.append(clean(d['gls_ref'].upper() if lower else d['gls_ref']))
            kp = np.array(corr[n]['keypoints']); conf.append(float(kp[:, HANDS, 2].mean()))
            names.append(n)
        conf = np.array(conf)
        w_video = wer(v_hyp, refs); w_fixed = wer(f_hyp, refs)
        # oracle clip switch: per clip pick lower-WER hyp
        oracle_hyp = [v_hyp[i] if wer([v_hyp[i]], [refs[i]]) <= wer([f_hyp[i]], [refs[i]]) else f_hyp[i]
                      for i in range(len(refs))]
        w_oracle = wer(oracle_hyp, refs)
        # realizable threshold switch: conf<thr -> video else fixed; best thr
        best_thr_wer, best_thr = 1e9, None
        for thr in np.quantile(conf, np.linspace(0, 1, 21)):
            hyp = [v_hyp[i] if conf[i] < thr else f_hyp[i] for i in range(len(refs))]
            w = wer(hyp, refs)
            if w < best_thr_wer: best_thr_wer, best_thr = w, float(thr)
        b = baseline.get(sev, {})
        row = dict(severity=sev, video_only=round(w_video, 2), fixed_fusion=round(w_fixed, 2),
                   co_training=b.get('co_training'), oracle_switch=round(w_oracle, 2),
                   thresh_switch=round(best_thr_wer, 2), best_thr=round(best_thr, 3))
        rows.append(row)
        print(f'[{args.ctype} sev={sev}] video={w_video:.2f} fixed={w_fixed:.2f} '
              f'cotrain={b.get("co_training")} ORACLE_switch={w_oracle:.2f} THRESH_switch={best_thr_wer:.2f}', flush=True)

    with open(os.path.join(args.out_dir, f'clip_switch_{args.ctype}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(os.path.join(args.out_dir, f'clip_switch_{args.ctype}.json'), 'w'), indent=2)
    print('RESULT:', json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
