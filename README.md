# OCC NeRF
## Quick Start
```bash
python tasks/homo/train_maskDL_long.py --base_dir ./data/LLFF/ --scene_name fern/images --resize_ratio 8 --gpu_id 0 --alias debug --pose_lr 3e-3  --focal_lr 1e-4 --nerf_lr 3e-4 --fx_only True --num_rows_eval_img 6 --epoch 100 --eval_interval 1
```