gpu=0
model=punet
extra_tag=punet_baseline3
epoch=99

mkdir outputs/${extra_tag}

#source ~/anaconda3/bin/activate torch12

python -u test.py \
    --model ${model} \
    --save_dir outputs/${extra_tag} \
    --gpu ${gpu} \
    --resume logs/${extra_tag}/punet_epoch_${epoch}.pth
