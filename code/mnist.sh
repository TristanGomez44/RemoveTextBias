python trainVal.py -c model.config --exp_id mnist --model_id cnn     --epochs 200 --dataset MNIST --model cnn
#The batch size of the gnn_resnet_mc has to be reduced because of a bug in pytorch
python trainVal.py -c model.config --exp_id mnist --model_id gnn     --epochs 200 --dataset MNIST --model gnn_resnet_mc --batch_size 32 --test_batch_size 32
python trainVal.py -c model.config --exp_id mnist --model_id gcnn    --epochs 200 --dataset MNIST --model gcnn_resnet
