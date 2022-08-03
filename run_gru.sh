module load wandb

python main.py --num_epochs 50 --rnn_bi 1 --use_ln yes --use_bn yes
python main.py --num_epochs 50 --rnn_bi 1 --use_ln no --use_bn no
python main.py --num_epochs 50 --rnn_bi 1 --use_ln yes --use_bn no

python main.py --num_epochs 50 --rnn_bi 2 --use_ln yes --use_bn yes
python main.py --num_epochs 50 --rnn_bi 2 --use_ln no --use_bn no
python main.py --num_epochs 50 --rnn_bi 2 --use_ln yes --use_bn no

read varname