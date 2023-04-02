
#!/bin/bash

#ELU alpha
# python main.py \
# -batch_size 64 \
# -dropout_rate 0.25 \
# -activation_function 'elu' \
# -elu_alpha 1.0

# python main.py \
# -batch_size 64 \
# -dropout_rate 0.25 \
# -activation_function 'elu' \
# -elu_alpha 0.6

# python main.py \
# -batch_size 64 \
# -dropout_rate 0.25 \
# -activation_function 'elu' \
# -elu_alpha 0.2

# python main.py \
# -batch_size 64 \
# -dropout_rate 0.25 \
# -activation_function 'elu' \
# -elu_alpha 1.4

# python main.py \
# -batch_size 64 \
# -dropout_rate 0.25 \
# -activation_function 'elu' \
# -elu_alpha 1.8


# Dropout
python main.py \
-num_epochs 150 \
-batch_size 64 \
-lr 0.001 \
-dropout_rate 0.5 \
-activation_function 'elu' \
-elu_alpha 1.0

python main.py \
-num_epochs 150 \
-batch_size 64 \
-lr 0.001 \
-dropout_rate 0.25 \
-activation_function 'elu' \
-elu_alpha 1.0

python main.py \
-num_epochs 150 \
-batch_size 64 \
-lr 0.001 \
-dropout_rate 0.75 \
-activation_function 'elu' \
-elu_alpha 1.0

# ReLu, ELU, LeakyReLu
python main.py \
-num_epochs 150 \
-batch_size 64 \
-lr 0.001 \
-dropout_rate 0.25 \
-activation_function 'elu' \
-elu_alpha 1.0

python main.py \
-num_epochs 150 \
-batch_size 64 \
-lr 0.001 \
-dropout_rate 0.25 \
-activation_function 'relu' \
-elu_alpha 1.0

python main.py \
-num_epochs 150 \
-batch_size 64 \
-lr 0.001 \
-dropout_rate 0.25 \
-activation_function 'leakyrelu' \
-elu_alpha 1.0






