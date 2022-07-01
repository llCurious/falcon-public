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
	if (PRE_LOAD)
	{
		network += " preloaded";
		PRELOADING = true;
		preload_network(PRELOADING, network, dataset, net);
	}

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
	// runTest("Debug", "BoolAnd", network);
	// runTest("Debug", "PosWrap", network);
	// runTest("Debug", "WCExtension", network);
	// runTest("Debug", "MixedShareGen", network);
	// runTest("Debug", "MSExtension", network);
	// runTest("Debug", "ProbTruncation", network);
	// runTest("Debug", "TruncAndReduce", network);
	// runTest("Debug", "RandBit", network);
	// runTest("Debug", "Reciprocal", network);
	// runTest("Debug", "InverseSqrt", network);
	// runTest("Debug", "funcDivisionByNR", network);
	// runTest("Debug", "BNLayer", network);
	// runTest("Bench", "MSExtension", network);
	// runTest("Bench", "WCExtension", network);
	// runTest("Bench", "BN", network);
	// runTest("Bench", "Trunc", network);
	// runTest("Bench", "SoftMax", network);
	// runTest("Debug", "Square", network);
	// runTest("Debug", "Exp", network);
	// runTest("Debug", "Softmax", network);
	// runTest("Debug", "BNLayer", network);
	// runTest("Test", "ReLUPrime1", network);

	// runTest("Test", "BN", network);
	// runTest("Test", "Division", network);
	// runTest("Test", "Maxpool", network);
	// runTest("Test", "Mat-Mul", network);
	// runTest("Test", "DotProd", network);

	// Run forward/backward for single layers
	//  1. what {F, D, U}
	// 	2. l {0,1,....NUM_LAYERS-1}
	// size_t l = 1;
	// string what = "F";
	// size_t count = 10;
	// // runOnly(net, l, what, network);
	// runOnlyLayer(net, 1, network, count);
	// for (size_t i = 0; i < 3; i++)
	// {
	// 	// from layer 0 -> 2
	// 	// time
	// 	auto start = std::chrono::system_clock::now();
	// 	runOnlyLayer(net, i, network, count);
	// 	auto end = std::chrono::system_clock::now();
	// 	double time = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6) / count;

	// 	// comm
	// 	uint64_t round = commObject.getRoundsRecv();
	// 	uint64_t commsize = commObject.getRecv();
	// 	runOnlyLayer(net, i, network, 1);
	// 	cout << i << OFFLINE_ON << " time: " << time << "    round: " << commObject.getRoundsRecv() - round << "    size: " << commObject.getRecv() - commsize << endl;
	// }

#if (!DEBUG_ONLY)
	// Run training
	// cout << "----------------------------------------------" << endl;
	// cout << "-------------------Run Training---------------" << endl;
	network += " train";
	printNetwork(net);
	train(net, network, dataset);

	// string type = MP_TRAINING ? "mix" : "high";
	// string off = OFFLINE_ON ? "all" : "on";
	// cout << network << " " << dataset << " inf " << NUM_ITERATIONS << " " << off << " " << type << endl;
	// network += " train";

	// uint64_t round = commObject.getRoundsRecv();
	// uint64_t commsize = commObject.getRecv();
	// auto start = std::chrono::system_clock::now();

	// train(net, network, dataset);

	// auto end = std::chrono::system_clock::now();
	// double time = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6);
	// cout << " time: " << time << "    round: " << commObject.getRoundsRecv() - round << "    size: " << commObject.getRecv() - commsize << endl;

	// Run inference (possibly with preloading a network)
	// cout << "----------------------------------------------" << endl;
	// cout << "-------------------Run Inference---------------" << endl;
	// test(PRELOADING, network, net, 2);
	// string type = MP_TRAINING ? "mix" : "high";
	// string off = OFFLINE_ON ? "all" : "on";
	// cout << network << " " << dataset << " inf " << NUM_ITERATIONS << " " << off << " " << type << endl;
	// network += " test";
	// // test time
	// int cnt = 2;
	// auto start = std::chrono::system_clock::now();
	// train(net, network, dataset);
	// // test(PRELOADING, network, net, cnt);
	// auto end = std::chrono::system_clock::now();
	// double time = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6) / cnt;
	// // comm
	// uint64_t round = commObject.getRoundsRecv();
	// uint64_t commsize = commObject.getRecv();
	// train(net, network, dataset);
	// // test(PRELOADING, network, net, 1);
	// cout << " time: " << time << "    round: " << commObject.getRoundsRecv() - round << "    size: " << commObject.getRecv() - commsize << endl;
	// // cout << cnt << " " << time << endl;
	end_m(network);
	// string type = MP_TRAINING ? "mix" : "high";
	// cout << network << " on " << dataset << " dataset " << (1<<LOG_MINI_BATCH) << " " << type << endl;
	cout << "----------------------------------------------" << endl;
	cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum
		 << "), " << NUM_ITERATIONS << " iterations, batch size " << MINI_BATCH_SIZE << endl
		 << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
	cout << "----------------------------------------------" << endl
		 << endl;
#endif

	/****************************** CLEAN-UP ******************************/
	// delete aes_indep;
	// delete aes_next;
	// delete aes_prev;
	delete config;
	delete net;
	deleteObjects();

	return 0;
}
