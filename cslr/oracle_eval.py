import argparse, os, sys, json
import numpy as np
import torch

sys.path.append(os.getcwd())
from modelling.model import build_model
from utils.misc import load_config, neq_load_customized, make_logger
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from utils.metrics import wer_list
from train_gate import load_pair, fuse
from train_gate_ds import build_alpha_targets


@torch.no_grad()
def oracle_wer(items, decoder, beam, clean, device):
    lower = decoder.gloss_tokenizer.lower_case
    hyps, refs = [], []
    for it in items:
        lv, lk = it['lv'].to(device), it['lk'].to(device)
        a = it['alpha_target'].to(device)                     # (T,) per-frame oracle
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
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='oracle.log')
    device = torch.device('cuda')

    cfg = load_config(args.decoder_config); cfg['device'] = device
    decoder = build_model(cfg)
    sd = torch.load(os.path.join(cfg['training']['model_dir'], 'ckpts', args.ckpt_name), map_location='cuda')
    neq_load_customized(decoder, sd['model_state'], verbose=False); decoder.eval()
    dsname = cfg['data']['dataset_name'].lower()
    clean = clean_phoenix_2014_trans if dsname == 'phoenix-2014t' else clean_phoenix_2014

    out = {}
    for split in ['dev', 'test']:
        items = load_pair(args.emis_dir, split)
        build_alpha_targets(items, args.fixed_alpha)
        wer = oracle_wer(items, decoder, args.beam, clean, device)
        out[f'oracle_{split}_wer'] = wer
        print(f'[oracle] {split} WER = {wer:.2f}', flush=True)
    json.dump(out, open(os.path.join(args.out_dir, 'oracle_result.json'), 'w'), indent=2)
    print('RESULT: ' + json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
