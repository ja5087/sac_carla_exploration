# python main.py -info sac_hopper -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 
# python main.py -info redq_hopper -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method redq -target_entropy -1.0
# CUDA_VISIBLE_DEVICES=0 python main.py -info droq_hopper -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.0001 -layer_norm 1 &

CUDA_VISIBLE_DEVICES=4 python main.py -info sac_walker -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 
# python main.py -info redq_walker -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method redq -target_entropy -1.0
# python main.py -info droq_walker -env Walker2d-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1

# python main.py -info sac_ant -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 
# python main.py -info redq_ant -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method redq -target_entropy -1.0
# CUDA_VISIBLE_DEVICES=1 python main.py -info droq_ant -env Ant-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.01 -layer_norm 1 &

# python main.py -info sac_humanoid -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 
# python main.py -info redq_humanoid -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method redq -target_entropy -1.0
# CUDA_VISIBLE_DEVICES=2 python main.py -info droq_humanoid -env Humanoid-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.1 -layer_norm 1 &
