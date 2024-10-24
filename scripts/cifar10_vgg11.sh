python main_train.py \
        --dataset cifar10 \
        --model vgg11 \
        --encode ft \
        --epoch 200 \
        --optim sgd \
        --lr 0.1\
        --tau 1.0\
        -b 256 \
        -T 4
