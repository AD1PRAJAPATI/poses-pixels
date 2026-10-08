import argparse, os, sys, pickle
import numpy as np
import torch
import torch.distributed as dist

sys.path.append(os.getcwd())


def init_single_process_group():
    """build_dataloader wraps the train split in a DistributedSampler, which
    needs a process group. Initialize a 1-process gloo group so it works
    outside torch.distributed.launch."""
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29566')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('LOCAL_RANK', '0')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
from modelling.model import build_model
from dataset.Dataloader import build_dataloader
from utils.misc import load_config, move_to_device, neq_load_customized, make_logger

PRIORITY = ['gloss_logits', 'rgb_gloss_logits', 'keypoint_gloss_logits',
            'fuse_gloss_logits', 'ensemble_last_gloss_logits']


def load_stream(config_path, ckpt_name):
    cfg = load_config(config_path)
    cfg['device'] = torch.device('cuda')
    cfg['training']['batch_size'] = 1
    model = build_model(cfg)
    p = os.path.join(cfg['training']['model_dir'], 'ckpts', ckpt_name)
    sd = torch.load(p, map_location='cuda')
    neq_load_customized(model, sd['model_state'], verbose=False)
    model.eval()
    print(f'loaded {p}', flush=True)
    return model, cfg


def pick_key(fo):
    for k in PRIORITY:
        if k in fo and fo[k] is not None:
            return k
    for k in fo:
        if 'gloss_logits' in k and fo[k] is not None:
            return k
    raise KeyError('no gloss_logits: ' + str(list(fo.keys())))


def to_ids(model, gls_ref):
    """Tokenize one gloss string to a list of target ids (best-effort)."""
    try:
        ri = model.gloss_tokenizer([gls_ref])
        lab = ri['gloss_labels']
        if isinstance(lab, torch.Tensor):
            lab = lab.flatten().tolist()
        else:
            lab = list(np.array(lab).flatten())
        return [int(x) for x in lab if int(x) != 0]   # drop pad/blank(0)
    except Exception as e:
        print('tokenize failed:', e, flush=True)
        return None


@torch.no_grad()
def dump(model, cfg, split, out_path):
    dl, _ = build_dataloader(cfg, split, model.text_tokenizer, model.gloss_tokenizer)
    out, key = {}, None
    for batch in dl:
        batch = move_to_device(batch, cfg['device'])
        fo = model(is_train=False, **batch)
        key = key or pick_key(fo)
        lg = fo[key]                                   # (1, T, V)
        il = fo['input_lengths']
        name = batch['name'][0]
        gls = batch['gloss'][0]
        out[name] = dict(
            logp=torch.log_softmax(lg, dim=-1)[0].half().cpu().numpy(),
            length=int(il[0].item()),
            gloss_labels=to_ids(model, gls),
            gls_ref=gls)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'wrote {out_path}: {len(out)} clips, key={key}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_config', required=True)
    ap.add_argument('--keypoint_config', required=True)
    ap.add_argument('--ckpt_name', default='best.ckpt')
    ap.add_argument('--out_dir', default='results/emissions')
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_logger(model_dir=args.out_dir, log_file='dump.log')
    init_single_process_group()

    for stream, cfgp in [('video', args.video_config), ('keypoint', args.keypoint_config)]:
        model, cfg = load_stream(cfgp, args.ckpt_name)
        for split in args.splits:
            dump(model, cfg, split, os.path.join(args.out_dir, f'emis_{stream}_{split}.pkl'))
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
