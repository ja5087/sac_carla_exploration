CUDA_VISIBLE_DEVICES=0 python main.py -info sac_spec_hopper -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -spectral_norm 1 &
CUDA_VISIBLE_DEVICES=1 python main.py -info sac_drop_spec_hopper -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -spectral_norm 1 & 
CUDA_VISIBLE_DEVICES=2 python main.py -info sac_drop_ln_spec_hopper -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1 -spectral_norm 1 &

CUDA_VISIBLE_DEVICES=3 python main.py -info sac_spec_walker -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -spectral_norm 1 &
CUDA_VISIBLE_DEVICES=4 python main.py -info sac_spec_ant -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -spectral_norm 1 &
CUDA_VISIBLE_DEVICES=5 python main.py -info sac_spec_humanoid -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -spectral_norm 1 &

CUDA_VISIBLE_DEVICES=6 python main.py -info sac_drop_ln_spec_walker -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1 -spectral_norm 1 &
CUDA_VISIBLE_DEVICES=7 python main.py -info sac_drop_ln_spec_ant -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.01 -layer_norm 1 -spectral_norm 1
CUDA_VISIBLE_DEVICES=0 python main.py -info sac_drop_ln_spec_humanoid -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.1 -layer_norm 1 -spectral_norm 1
