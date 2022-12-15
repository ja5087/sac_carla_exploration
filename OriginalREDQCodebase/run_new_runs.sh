# # Ant 
# # -> L1
# CUDA_VISIBLE_DEVICES=0 python main.py -info sac_l1_0_5_ant -l1 0.5 -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
# CUDA_VISIBLE_DEVICES=1 python main.py -info sac_l1_0_05_ant -l1 0.05 -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
# CUDA_VISIBLE_DEVICES=2 python main.py -info sac_l1_0_005_ant -l1 0.005 -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &

# # -> drop_spec
# CUDA_VISIBLE_DEVICES=3 python main.py -info sac_drop_spec_ant -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -spectral_norm 1 &

# # Hopper
# # -> L1
# CUDA_VISIBLE_DEVICES=4 python main.py -info sac_l1_0_5_hopper -l1 0.5 -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
# CUDA_VISIBLE_DEVICES=5 python main.py -info sac_l1_0_05_hopper -l1 0.05 -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
# CUDA_VISIBLE_DEVICES=6 python main.py -info sac_l1_0_005_hopper -l1 0.005 -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &

# # Humanoid
# # -> L1
# CUDA_VISIBLE_DEVICES=7 python main.py -info sac_l1_0_5_humanoid -l1 0.5 -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
# CUDA_VISIBLE_DEVICES=0 python main.py -info sac_l1_0_05_humanoid -l1 0.05 -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
# CUDA_VISIBLE_DEVICES=1 python main.py -info sac_l1_0_005_humanoid -l1 0.005 -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &

# -> drop_spec
# CUDA_VISIBLE_DEVICES=2 python main.py -info sac_drop_spec_humanoid -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -spectral_norm 1 &

# Walker
# -> L1
# CUDA_VISIBLE_DEVICES=3 python main.py -info sac_l1_0_5_walker -l1 0.5 -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
# CUDA_VISIBLE_DEVICES=4 python main.py -info sac_l1_0_05_walker -l1 0.05 -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
# CUDA_VISIBLE_DEVICES=5 python main.py -info sac_l1_0_005_walker -l1 0.005 -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &

# -> drop_spec
# CUDA_VISIBLE_DEVICES=6 python main.py -info sac_drop_spec_walker -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -spectral_norm 1 &

# -> droq
# CUDA_VISIBLE_DEVICES=4 python main.py -info sac_droq_walker -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method redq -target_entropy -1.0 &

# -> redq
# CUDA_VISIBLE_DEVICES=5 python main.py -info sac_redq_walker -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1 &

# Ant L2
CUDA_VISIBLE_DEVICES=4 python main.py -info sac_l2_0_001_ant -l2 0.001 -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
CUDA_VISIBLE_DEVICES=5 python main.py -info sac_l2_0_1_ant -l2 0.1 -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &

# Hopper L2
CUDA_VISIBLE_DEVICES=4 python main.py -info sac_l2_0_001_hopper -l2 0.001 -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
CUDA_VISIBLE_DEVICES=5 python main.py -info sac_l2_0_1_hopper -l2 0.1 -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &

# Humanoid L2
CUDA_VISIBLE_DEVICES=4 python main.py -info sac_l2_0_001_humanoid -l2 0.001 -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
CUDA_VISIBLE_DEVICES=5 python main.py -info sac_l2_0_1_humanoid -l2 0.1 -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &

# Walker L2
CUDA_VISIBLE_DEVICES=4 python main.py -info sac_l2_0_001_walker -l2 0.001 -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &
CUDA_VISIBLE_DEVICES=5 python main.py -info sac_l2_0_1_walker -l2 0.1 -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 &