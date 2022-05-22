#include <iostream>
#include <string>
#include "AESObject.h"
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "unitTests.h"

int partyNum;
// AESObject* aes_indep;
// AESObject* aes_next;
// AESObject* aes_prev;
Precompute *PrecomputeObject;

int main(int argc, char **argv)
{
	/****************************** PREPROCESSING ******************************/
	parseInputs(argc, argv);
	NeuralNetConfig *config = new NeuralNetConfig(NUM_ITERATIONS);
	string network, dataset, security;
	bool PRELOADING = false;

	/****************************** SELECT NETWORK ******************************/
	// Network {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}
	// Dataset {MNIST, CIFAR10, and ImageNet}
	// Security {Semi-honest or Malicious}
	if (argc == 9)
	{
		network = argv[6];
		dataset = argv[7];
		security = argv[8];
	}
	else
	{
		network = "SecureML";
		dataset = "MNIST";
		security = "Semi-honest";
	}
	selectNetwork(network, dataset, security, config);
	config->checkNetwork();
	NeuralNetwork *net = new NeuralNetwork(config);

	/****************************** AES SETUP and SYNC ******************************/
	// aes_indep = new AESObject(argv[3]);
	// aes_next = new AESObject(argv[4]);
	// aes_prev = new AESObject(argv[5]);

	initializeCommunication(argv[2], partyNum);
	synchronize(2000000);

	PrecomputeObject = new Precompute(partyNum, "files/key");

	/****************************** RUN NETWORK/UNIT TESTS ******************************/
	// Run these if you want a preloaded network to be tested
	// assert(NUM_ITERATION == 1 and "check if readMiniBatch is false in test(net)")
	// First argument {SecureML, Sarda, MiniONN, or LeNet}
	//  network += " preloaded"; PRELOADING = true;
	//  preload_network(PRELOADING, network, net);

	start_m();
	// Run unit tests in two modes:
	//	1. Debug {Mat-Mul, DotProd, PC, Wrap, ReLUPrime, ReLU, Division, BN, SSBits, SS, and Maxpool}
	//	2. Test {Mat-Mul1, Mat-Mul2, Mat-Mul3 (and similarly) Conv*, ReLU*, ReLUPrime*, and Maxpool*} where * = {1,2,3}
	//  runTest("Debug", "DotProd", network);
	// runTest("Debug", "BN", network);
	// runTest("Debug", "Division", network);
	// runTest("Debug", "Maxpool", network);
	// runTest("Debug", "Mat-Mul", network);
	// runTest("Debug", "DotProd", network);
	// runTest("Debug", "Reduction", network);
	// runTest("Debug", "PairRandom", network);
	// runTest("Debug", "PartyShare", network);
	// runTest("Debug","ZeroRandom",network);
	runTest("Debug", "BoolAnd", network);
	// runTest("Debug", "PosWrap", network);
	// runTest("Debug", "WC-Extension", network);
	runTest("Debug", "Square", network);
	runTest("Debug", "Exp", network);
	runTest("Debug", "Softmax", network);
	// runTest("Test", "ReLUPrime1", network);

	// runTest("Test", "BN", network);
	// runTest("Test", "Division", network);
	// runTest("Test", "Maxpool", network);
	// runTest("Test", "Mat-Mul", network);
	// runTest("Test", "DotProd", network);

	// Run forward/backward for single layers
	//  1. what {F, D, U}
	// 	2. l {0,1,....NUM_LAYERS-1}
	// size_t l = 0;
	// string what = "F";
	// runOnly(net, l, what, network);

#if (!DEBUG_ONLY)
	// Run training
	network += " train";
	train(net);

	// Run inference (possibly with preloading a network)
	//  network += " test";
	//  test(PRELOADING, network, net);

	end_m(network);
	cout << "----------------------------------------------" << endl;
	cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum
		 << "), " << NUM_ITERATIONS << " iterations, batch size " << MINI_BATCH_SIZE << endl
		 << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
	cout << "----------------------------------------------" << endl
		 << endl;
#endif

	// printNetwork(net);

	/****************************** CLEAN-UP ******************************/
	// delete aes_indep;
	// delete aes_next;
	// delete aes_prev;
	delete config;
	delete net;
	deleteObjects();

	return 0;
}
