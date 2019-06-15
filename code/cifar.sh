python trainVal.py -c model.config --exp_id cifar --model_id cnn           --epochs 200 --dataset CIFAR10 --model cnn
python trainVal.py -c model.config --exp_id cifar --model_id gnn_resnet    --epochs 200 --dataset CIFAR10 --model gnn_resnet
#The batch size of the gnn_resnet_mc has to be reduced because of a bug in pyto
python trainVal.py -c model.config --exp_id cifar --model_id gnn_resnet_mc --epochs 200 --dataset CIFAR10 --model gnn_resnet_mc --batch_size 32 --test_batch_size 32
python trainVal.py -c model.config --exp_id cifar --model_id gcnn_resnet   --epochs 200 --dataset CIFAR10 --model gcnn_resnet
