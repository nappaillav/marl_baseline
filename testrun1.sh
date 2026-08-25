#!/bin/bash

python src/main.py --config=hpn_saleq_sem_wm_qmix --env-config=sc2_v2_protoss with obs_agent_id=True obs_last_action=False runner=parallel buffer_size=4000 t_max=5005000 save_model=False use_cuda=True use_tensorboard=True
