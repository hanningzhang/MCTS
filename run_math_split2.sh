CUDA_VISIBLE_DEVICES=0 bash mc_reward_data_gsm.sh 0 2 &
CUDA_VISIBLE_DEVICES=1 bash mc_reward_data_gsm.sh 1 2 &
CUDA_VISIBLE_DEVICES=2 bash mc_reward_data_gsm.sh 2 2 &
CUDA_VISIBLE_DEVICES=3 bash mc_reward_data_gsm.sh 3 2
