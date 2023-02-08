dataset=$1
echo "Load "$dataset

if [ "$dataset" == "mnist" ];then
    make mnist
    ./mnist ~/Downloads/mnist_train.csv ~/Downloads/mnist_test.csv
fi

if [ "$dataset" == "cifar" ]; then
    make cifar
    python CIFAR10Loader.py
    ./cifar ~/Downloads/cifar10_train.csv ~/Downloads/cifar10_test.csv
fi

if [ "$dataset" == "imagenet" ]; then
    make imagenet
    python TinyImageNetLoader.py
    ./imagenet ~/Downloads/imagenet_train.csv ~/Downloads/imagenet_test.csv
fi

make clean