
#data='/home/anilkag/datasets/imagenet-1000/'
#data='/Data/anil_datasets/imagenet-1000/'
data='/mnt/active/datasets/imagenet/'

#strategy='KD'
strategy='DiSK'

lmbda_adaptive=5 
#lmbda_adaptive=1 

topK=20 #50
#routing='ImageNet_Routing_B1'
routing='ImageNet_Routing_B2'
#routing='ImageNet_Routing_B3'

#student='vit_tiny_patch16_224'
#teacher='vit_large_patch16_224'

student='deit_tiny_patch16_224'
teacher='vit_large_patch16_224'
#teacher='regnety_160'

#student='resnet18'
#teacher='resnet50'

#teacher='tf_efficientnet_b2' #'resnet50'
#teacher='mobilenetv3_large_100' #'tf_efficientnet_b0' #'resnet50'
#teacher='tf_mobilenetv3_large_100' #'tf_efficientnet_b0' #'resnet50'

max_ce=2.0
alpha=0.5
tmp_t=1.  #4.0
tmp_s=1. #2.0

#ckpt='./output/train/20230217-104200-vit_tiny_patch16_224-224/model_best.pth.tar'
#ckpt='./output/train/20230217-104200-vit_tiny_patch16_224-224/last.pth.tar'
#ckpt='./output/train/20230221-180131-vit_tiny_patch16_224-224/last.pth.tar'

CUDA_VISIBLE_DEVICES='0,1,2,3' ./distributed_train.sh 4 $data -b 64 \
	--model $student --teacher $teacher --routing $routing -j 16 \
        --opt adamw  --warmup-lr 1e-6 \
	--sched cosine --epochs 90 --lr 1e-4 --amp --weight-decay 5e-4 \
        --strategy $strategy --temperature_t $tmp_t --temperature_s $tmp_s \
	--budget_g 0.2 --budget_g_max 0.3 --budget_g_min 0.1 --budget_Ti 30 \
	--lmbda .5 --lmbda_min 0.1 --Ti 30 --topK $topK --max_ce $max_ce \
	--model-ema --model-ema-decay 0.9999 --pretrained --alpha $alpha \
	--aa rand-m9-mstd0.5-inc1  #--sync-bn #--resume $ckpt


#	--sched cosine --epochs 90 --lr 0.24 --amp --weight-decay 5e-5 \
