export PYTHONPATH="/rscratch/ja/carla/CARLA_SIM/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg:$PYTHONPATH"
python train.py --xparl_addr localhost:8080 --train_total_steps 500000 --test_every_steps 1000 --algorithm droq --critic_count 2 --dropout_p 0.01 --utd 20 --results_dir results/droq_a1
