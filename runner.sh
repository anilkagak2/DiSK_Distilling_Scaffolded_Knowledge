GPU='0'

dataset='cifar100'
data_path='/home/anilkag/code/data/cifar/'
#pretrained_models_dir='/home/anilkag/code/github/Knowledge-Distillation-Pvt/gold_models'
pretrained_models_dir='/home/anilkag/code/github/DiSK_Distilling_Scaffolded_Knowledge/pretrained-models/'

kt=3
kg=3
epochs=200
alt_epochs=40
init_epochs=5

#rt='b26_default'
#rt='b25_default'
rt='b24_default'
#rt='b23_default'

aux=0.1
nll=0.9
kl=0.0
oracle=0.0
use_aux=1
use_kl=0
use_oracle=0

s_iters=250
g_iters=250
eval_batch_size=200
batch_size=200
use_alt_min=1

b_add_sparsity_alt_min=0
KD_temperature_t=4.0
KD_temperature_s=4.0
topK=10 
max_ce=5.

lmbda=10.
lmbda_dual=15.
lmbda_min=1.
lmbda_Ti=50

budget_g=0.6
budget_g_min=0.2
budget_g_max=0.4
budget_Ti=20

ckpt='""'

# 1159M MACs, 80.46%
#global_name='ResNet34' 
#global_ckpt="${pretrained_models_dir}/disk-CE-cifar100-ResNet34-model_best.pth.tar"

# 555M MACs, 76.56%
#global_name='ResNet18' 
#global_ckpt="${pretrained_models_dir}/disk-CE-cifar100-ResNet18-model_best.pth.tar"

# 253M MACs, 75.25%
#global_name='ResNet10' 
#global_ckpt="${pretrained_models_dir}/disk-CE-cifar100-ResNet10-model_best.pth.tar"

# 64M MACs, 71.99%
global_name='ResNet10_l'
global_ckpt="${pretrained_models_dir}/disk-CE-cifar100-ResNet10_l-model_best.pth.tar"


# 0.8M MACs, 28.21%
#base_name='ResNet10_xxxs'
#base_ckpt="${pretrained_models_dir}/disk-CE-cifar100-ResNet10_xxxs-model_best.pth.tar"

# 2M MACs, 32.05%
#base_name='ResNet10_xxs'
#base_ckpt="${pretrained_models_dir}/disk-CE-cifar100-ResNet10_xxs-model_best.pth.tar"

# 2.86M MACs, 42.99%
#base_name="ResNet10_xs"
#base_ckpt="${pretrained_models_dir}/disk-CE-cifar100-ResNet10_xs-model_best.pth.tar"

# 4M MACs, 52.16%
base_name="ResNet10_s"
base_ckpt="${pretrained_models_dir}/disk-CE-cifar100-ResNet10_s-model_best.pth.tar"

# 16M MACs, 65.24%
#base_name='ResNet10_m'
#base_ckpt="${pretrained_models_dir}/disk-CE-cifar100-ResNet10_m-model_best.pth.tar"
#ckpt="${pretrained_models_dir}/disk-hints10-cifar100-ResNet10_m-b24_default-ResNet10_l-1-23-0-40-0-1-0-1-0.1-0.1-1e-05-sgd-sgd-hybrid_kd_inst-0.9-4.0-50-20-0.2_best_model.pth.tar"

b_add_sparsity_alt_min=0
penalty=24 #25
st=50 #42

topK=10 #5 #10 
max_ce=5.

lmbda_dual=5.
lmbda=5.
lmbda_min=1.
lmbda_adaptive=5

KD_temperature_s=3.5
budget_g_min=0.4
budget_g_max=0.6

#eval_batch_size=32
#batch_size=32

echo "T=$global_name, S=$base_name"
CUDA_VISIBLE_DEVICES="$GPU"  python train_DiSK.py --dataset $dataset --data_path $data_path \
      	--model_name $base_name  --global_model_name $global_name  \
	--model_ckpt $base_ckpt \
	--global_model_ckpt $global_ckpt  --eval_batch_size $eval_batch_size --batch_size $batch_size  \
	--s_iters $s_iters --g_iters $g_iters --strategy 1 --s_lr 0.1 --g_lr 0.1 --g_lr_init 0.1 --kt $kt --kg $kg \
        --routing_opt_type "sgd" --base_opt_type "sgd"  --eps 1. \
	--base_method "hybrid_kd_inst"  --KD_temperature $KD_temperature_t \
        --KD_temperature_s $KD_temperature_s --topK $topK --max_ce $max_ce \
        --use_auxiliary $use_aux --penalty $penalty  --use_kl $use_kl  --use_oracle $use_oracle  --use_prob $st \
        --l_aux $aux --l_kl $kl --l_oracle $oracle --l_nll $nll \
        --_ckpt $ckpt \
	--use_alt_min $use_alt_min --init_epochs $init_epochs  --epochs $epochs  --routing_name  $rt \
	--lmbda $lmbda --lmbda_min $lmbda_min --lmbda_adaptive $lmbda_adaptive --Ti $lmbda_Ti --lmbda_dual $lmbda_dual --rho 2.  \
       	--budget_g $budget_g --budget_g_min $budget_g_min --budget_g_max $budget_g_max --budget_Ti $budget_Ti \
        --primal_budget_update 0 --b_add_sparsity_alt_min $b_add_sparsity_alt_min


