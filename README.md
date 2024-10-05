# MCTS Getting Start
```
pip install -r requirements.txt
```
We have GSM-8K and MATH datasets. For each dataset, we split the data in half to accelerate the generation. There are 4 scripts for generation in total.

For each generation script, we assume there are 4 GPUs for the task.
Run
```
run_gsm_split0.sh
run_gsm_split1.sh
run_math_split0.sh
run_math_split1.sh
```
For each script, it generates MCTS data on 4 GPUs. And utilizing 16 GPUs for these 4 scripts would be fastest.
