import argparse, os, sys, json, csv, pickle
import numpy as np
import torch

sys.path.append(os.getcwd())
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from utils.misc import make_logger
from clip_switch_eval import decode_one, wer
from degrade_eval import load_model, corrupt_dev_keypoints, run_emis, S2G_PRIORITY
from train_gate_rel2 import HANDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s2g_config', required=True)
    ap.add_argument('--video_emis', required=True)
    ap.add_argument('--ctype', default='dropout')
    ap.add_argument('--sevs', type=float, nargs='+', default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--out_dir', default='results/meta_switch')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='meta_switch.log')
    device = torch.device('cuda')

    video_emis = pickle.load(open(args.video_emis, 'rb'))
    s2g, s2g_cfg = load_model(args.s2g_config)
    lower = s2g.gloss_tokenizer.lower_case
    clean = clean_phoenix_2014_trans if s2g_cfg['data']['dataset_name'].lower() == 'phoenix-2014t' else clean_phoenix_2014
    KP = pickle.load(open(s2g_cfg['data']['keypoint_file'], 'rb'))

    rows = []
    for sev in args.sevs:
        corr = corrupt_dev_keypoints(KP, args.ctype, sev)
        tf = os.path.join(args.out_dir, f'_kp_{sev}.pkl'); pickle.dump(corr, open(tf, 'wb'))
        s2g_emis = run_emis(s2g, s2g_cfg, tf, S2G_PRIORITY); os.remove(tf)
        c_hyp, v_hyp, refs, conf = [], [], [], []
        for n, d in s2g_emis.items():
            if n not in video_emis: continue
            lc = torch.from_numpy(d['logp']).to(device)
            lv = torch.from_numpy(video_emis[n]['logp']).to(device)
            c_hyp.append(clean(decode_one(torch.log_softmax(lc, -1), s2g, args.beam, device, lower)))
            v_hyp.append(clean(decode_one(torch.log_softmax(lv, -1), s2g, args.beam, device, lower)))
            refs.append(clean(d['gls_ref'].upper() if lower else d['gls_ref']))
            kp = np.array(corr[n]['keypoints']); conf.append(float(kp[:, HANDS, 2].mean()))
        conf = np.array(conf)
        w_cotrain = wer(c_hyp, refs); w_video = wer(v_hyp, refs)
        oracle = [v_hyp[i] if wer([v_hyp[i]], [refs[i]]) <= wer([c_hyp[i]], [refs[i]]) else c_hyp[i] for i in range(len(refs))]
        w_oracle = wer(oracle, refs)
        best_w, best_thr = 1e9, None
        for thr in np.quantile(conf, np.linspace(0, 1, 21)):
            hyp = [v_hyp[i] if conf[i] < thr else c_hyp[i] for i in range(len(refs))]
            w = wer(hyp, refs)
            if w < best_w: best_w, best_thr = w, float(thr)
        rows.append(dict(severity=sev, video_only=round(w_video, 2), co_training=round(w_cotrain, 2),
                         oracle_meta=round(w_oracle, 2), thresh_meta=round(best_w, 2), best_thr=round(best_thr, 3)))
        print(f'[{args.ctype} sev={sev}] video={w_video:.2f} cotrain={w_cotrain:.2f} '
              f'ORACLE_meta={w_oracle:.2f} THRESH_meta={best_w:.2f}', flush=True)

    with open(os.path.join(args.out_dir, f'meta_switch_{args.ctype}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(os.path.join(args.out_dir, f'meta_switch_{args.ctype}.json'), 'w'), indent=2)
    print('RESULT:', json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
