export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

# 单卡训练，无需 deepspeed launcher
python3 prm_train.py "$@"