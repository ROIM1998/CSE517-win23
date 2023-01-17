for cutoff in 10 20 50 100
do
    for lr in 0.05 0.1 0.2 0.3 0.5
    do
        python train.py $cutoff $lr
    done
done