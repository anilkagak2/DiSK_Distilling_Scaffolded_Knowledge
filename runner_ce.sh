GPU='0'

dataset='cifar100'
data_path='/home/anilkag/code/data/cifar/'
save_dir='./new_models'

batch_size=200
eval_batch_size=200

epochs=200
lr=0.1
wd=5e-4
momentum=0.9

# 4M MACs, 52.16%
base_name="ResNet10_s"

echo "S=$base_name"
CUDA_VISIBLE_DEVICES="$GPU"  python train_CE.py --dataset $dataset --data_path $data_path \
        --model_name $base_name --save_dir $save_dir \
	--eval_batch_size $eval_batch_size --batch_size $batch_size  \
	--lr $lr --wd $wd --momentum $momentum  


