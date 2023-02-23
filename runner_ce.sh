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

# 1159M MACs, 80.46%
#base_name='ResNet34' 

# 555M MACs, 76.56%
#base_name='ResNet18' 

# 253M MACs, 75.25%
#base_name='ResNet10' 

# 64M MACs, 71.99%
#base_name='ResNet10_l'

# 0.8M MACs, 28.21%
#base_name='ResNet10_xxxs'

# 2M MACs, 32.05%
#base_name='ResNet10_xxs'

# 2.86M MACs, 42.99%
#base_name="ResNet10_xs"

# 4M MACs, 52.16%
base_name="ResNet10_s"

# 16M MACs, 65.24%
#base_name='ResNet10_m'


echo "S=$base_name"
CUDA_VISIBLE_DEVICES="$GPU"  python train_CE.py --dataset $dataset --data_path $data_path \
        --model_name $base_name --save_dir $save_dir \
	--eval_batch_size $eval_batch_size --batch_size $batch_size  \
	--lr $lr --wd $wd --momentum $momentum --epochs $epochs 


