python mxnet/example/image-classification/fine-tune.py \
	--pretrained-model mobilenet \
	--load-epoch 0 \
	--gpus 0 \
	--data-train /data/train_data.rec \
	--data-val /data/test_data.rec \
	--batch-size 16 \
	--num-classes 2 \
	--num-examples 50000 \
	--model-prefix output/resnet-50
