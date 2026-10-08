# When Does Fusing Streams Help? Fragility of Multi-Stream Sign Recognition under Input Degradation

Code, degradation protocol, and analysis for the paper. Anonymized for double-blind review.
Set `export ROOT=/path/to/workdir` first. `cslr/` scripts go inside the public TwoStream-SLR
`TwoStreamNetwork/` directory; `islr/` scripts run against a pose Conformer + SignRep head.

## Degradation protocol
Operators act on the pose input at severity `s` and re-run the stream (appearance left clean):
frame dropout, hand occlusion, coordinate noise.

## CSLR (PHOENIX-2014) — run from TwoStreamNetwork/

Cache per-stream emissions:
```
python dump_emissions.py \
  --video_config experiments/configs/TwoStream/phoenix-2014_video.yaml \
  --keypoint_config experiments/configs/TwoStream/phoenix-2014_keypoint.yaml \
  --out_dir results/emissions --splits train dev test
```
Brittleness curves (baselines):
```
python degrade_eval.py \
  --keypoint_config experiments/configs/TwoStream/phoenix-2014_keypoint.yaml \
  --s2g_config experiments/configs/TwoStream/phoenix-2014_s2g.yaml \
  --video_emis results/emissions/emis_video_dev.pkl \
  --ctype dropout --sevs 0.0 0.1 0.25 0.5 0.75 0.9 --out_dir results/degrade
```
Clip-level reliability gate (co-training variant):
```
python meta_switch_eval.py \
  --s2g_config experiments/configs/TwoStream/phoenix-2014_s2g.yaml \
  --video_emis results/emissions/emis_video_dev.pkl \
  --ctype dropout --sevs 0.0 0.1 0.25 0.5 0.75 0.9 --out_dir results/meta_switch
```
Leakage-free evaluation (tau on dev, report on test, 3 seeds):
```
python final_eval_cslr.py \
  --keypoint_config experiments/configs/TwoStream/phoenix-2014_keypoint.yaml \
  --s2g_config experiments/configs/TwoStream/phoenix-2014_s2g.yaml \
  --emis_dir results/emissions --out_dir results/final \
  --ctypes dropout hand_occl --sevs 0.0 0.25 0.5 0.75 0.9 --seeds 0 1 2 --alpha 0.45
```
Per-frame-vs-clip analysis:
```
python oracle_eval.py --emis_dir results/emissions \
  --decoder_config experiments/configs/TwoStream/phoenix-2014_keypoint.yaml --fixed_alpha 0.45
python oracle_variants.py --emis_dir results/emissions \
  --decoder_config experiments/configs/TwoStream/phoenix-2014_keypoint.yaml \
  --fixed_alpha 0.45 --windows 3 5 9 15
python keypoint_signal_check.py \
  --keypoint_file ${ROOT}/data/phoenix-2014/phoenix-2014-keypoints.pkl \
  --emis_dir results/emissions --fit_split train --eval_split dev
```

## ISLR (ASL-Citizen) — run from the islr pipeline dir

Train the SignRep head:
```
python train_signrep_head.py --feat_dir signrep_feats \
  --train_csv ${ROOT}/data/asl_citizen/ASL_Citizen/splits/train.csv \
  --val_csv   ${ROOT}/data/asl_citizen/ASL_Citizen/splits/val.csv \
  --test_csv  ${ROOT}/data/asl_citizen/ASL_Citizen/splits/test.csv \
  --out_dir signrep_head --pool attn --arcface --epochs 60
```
Brittleness curves:
```
python islr_degrade.py --pose_ckpt checkpoints_islr/islr_base_seed42/best.pt \
  --pose_vocab checkpoints_islr/islr_base_seed42/vocab.json \
  --pose_feat_dir ${ROOT}/data/asl_citizen/pose_features \
  --signrep_head signrep_head/best.pt --signrep_feat_dir signrep_feats \
  --test_csv ${ROOT}/data/asl_citizen/ASL_Citizen/splits/test.csv \
  --ctype dropout --sevs 0.0 0.1 0.25 0.5 0.75 0.9
```
Leakage-free evaluation (tau on val, report on test, 3 seeds):
```
python islr_final.py --pose_ckpt checkpoints_islr/islr_base_seed42/best.pt \
  --pose_vocab checkpoints_islr/islr_base_seed42/vocab.json \
  --pose_feat_dir ${ROOT}/data/asl_citizen/pose_features \
  --signrep_head signrep_head/best.pt --signrep_feat_dir signrep_feats \
  --val_csv ${ROOT}/data/asl_citizen/ASL_Citizen/splits/val.csv \
  --test_csv ${ROOT}/data/asl_citizen/ASL_Citizen/splits/test.csv \
  --ctypes dropout hand_occl --sevs 0.0 0.25 0.5 0.75 0.9 --seeds 0 1 2
```

## Notes
Beam size 5 (CSLR) / top-1 accuracy (ISLR). The gate threshold and fusion weight are selected
on dev/val and applied unchanged at test; no test-set information enters model selection.
