# Install Python Environment
pip install -r requirements.txt
# Dataset Info

## MNIST
No header info
1. Download csv files from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
2. Delete the header info in csv files
3. bash run.sh mnist

## CIFAR-10
The test file is manually parsed according to https://www.cs.toronto.edu/~kriz/cifar.html. 

The input of CIFAR-10 over AlexNet is 227*227. https://datascience.stackexchange.com/questions/29245/what-is-the-input-size-of-alex-net
1. Download dataset from the website above. Move it to ~/Downloads
2. bash run.sh cifar

## Tiny ImageNet
The data is parsed according to https://github.com/pranavphoenix
1. Download dataset: http://cs231n.stanford.edu/tiny-imagenet-200.zip. Move it to ~/Downloads
1. bash run.sh imagenet