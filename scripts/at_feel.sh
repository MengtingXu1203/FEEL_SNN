python main_train.py \
        --dataset cifar100 \
        --model vgg11 \
        --encode ft \
        -m attrain \
        --epoch 300 \
        --optim sgd \
        --lr 0.1\
        --tau 1.0\
        -b 256 \
        -T 8
