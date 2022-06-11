
CXX=g++
RM=rm -f
FLAGS=-std=c++11

MNIST = mnist
CIFAR10 = cifar
ImageNet = imagenet

$(MNIST): MNISTParse.cpp MNISTParse.h globals.h
	$(CXX) $(FLAGS) -o $(MNIST) MNISTParse.cpp

$(CIFAR10): CIFAR10Parser.cpp CIFAR10Parser.h globals.h
	$(CXX) $(FLAGS) -o $(CIFAR10) CIFAR10Parser.cpp

$(ImageNet): ImageNetParser.cpp ImageNetParser.h globals.h
	$(CXX) $(FLAGS) -o $(ImageNet) ImageNetParser.cpp

clean:
	$(RM) $(MNIST)
	$(RM) $(CIFAR10)
	$(RM) $(ImageNet)

superclean:
	$(RM) mnist_*