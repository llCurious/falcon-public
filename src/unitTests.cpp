#include "Functionalities.h"

void debugPartySS()
{
	size_t size = 5;

	vector<highBit> a1(size);
	vector<lowBit> a2(size);
	for (size_t i = 0; i < size; i++)
	{
		a1[i] = rand();
		a2[i] = rand();
	}

	RSSVectorHighType test_result1(size);
	RSSVectorLowType test_result2(size);

	vector<highBit> result_plain1(size);
	vector<lowBit> result_plain2(size);

	// NUM_OF_PARTIES
	for (size_t i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i == partyNum)
		{
			funcShareSender<RSSVectorHighType, highBit>(test_result1, a1, size);
		}
		else
		{
			funcShareReceiver<RSSVectorHighType, highBit>(test_result1, size, i);
		}

		if (i == partyNum)
		{
			funcShareSender<RSSVectorLowType, lowBit>(test_result2, a2, size);
		}
		else
		{
			funcShareReceiver<RSSVectorLowType, lowBit>(test_result2, size, i);
		}

#if (!LOG_DEBUG)
		funcReconstruct<RSSVectorHighType, highBit>(test_result1, result_plain1, size, "high output", false);
		funcReconstruct<RSSVectorLowType, lowBit>(test_result2, result_plain2, size, "low output", false);
		assert(equalRssVector(result_plain1, a1, size));
		assert(equalRssVector(result_plain2, a2, size));
#endif
	}
}

void debugPairRandom()
{
	size_t size = 5;
	RSSVectorHighType test_result1(size);
	PrecomputeObject->getPairRand<RSSVectorHighType, highBit>(test_result1, size);

	printRssVector<RSSVectorHighType>(test_result1, "UNSIGNED", "pair random", size);
	// Precompute::getPairRand(Vec &a, size_t size)
}

void debugReduction()
{
	size_t size = 5;
	RSSVectorHighType test_data(size);
	RSSVectorLowType test_result(size);

	vector<highBit> a = {-5, 25, 0, 976, -916};

	funcGetShares(test_data, a);

	funcReduction(test_result, test_data);

	// cout << "begin print ss" << endl;

	// print_vector(test_data, "UNSIGNED", "test_data ss", test_data.size());
	// print_vector(test_result, "UNSIGNED", "test_result ss", test_result.size());

	vector<highBit> data_plain(size);
	vector<lowBit> result_plain(size);

	cout << "Reconstruct" << endl;

#if (!LOG_DEBUG)
	funcReconstruct<RSSVectorHighType, highBit>(test_data, data_plain, size, "input", true);
	funcReconstruct<RSSVectorLowType, lowBit>(test_result, result_plain, size, "output", true);
#endif

	for (size_t i = 0; i < size; i++)
	{
		// cout << data_plain[i] << " " << result_plain[i] << endl;
		assert((lowBit)(data_plain[i]) == (lowBit)(result_plain[i]));
	}
}

void debugZeroRandom()
{
	size_t size = 4;
	vector<highBit> a1(size);
	PrecomputeObject->getZeroShareRand<RSSVectorHighType, highBit>(a1, size);
	print_vector(a1, "UNSIGNED", "0 high bit", size);

	vector<lowBit> a2(size);
	PrecomputeObject->getZeroShareRand<RSSVectorLowType, lowBit>(a2, size);
	print_vector(a2, "UNSIGNED", "0 low bit", size);

}

void debugPosWrap()
{
}

void debugWCExtension()
{
}

void runTest(string str, string whichTest, string &network)
{
	if (str.compare("Debug") == 0)
	{
		if (whichTest.compare("Mat-Mul") == 0)
		{
			network = "Debug Mat-Mul";
			debugMatMul();
		}
		else if (whichTest.compare("DotProd") == 0)
		{
			network = "Debug DotProd";
			debugDotProd();
		}
		else if (whichTest.compare("PC") == 0)
		{
			network = "Debug PrivateCompare";
			debugPC();
		}
		else if (whichTest.compare("Wrap") == 0)
		{
			network = "Debug Wrap";
			debugWrap();
		}
		else if (whichTest.compare("ReLUPrime") == 0)
		{
			network = "Debug ReLUPrime";
			debugReLUPrime();
		}
		else if (whichTest.compare("ReLU") == 0)
		{
			network = "Debug ReLU";
			debugReLU();
		}
		else if (whichTest.compare("Division") == 0)
		{
			network = "Debug Division";
			debugDivision();
		}
		else if (whichTest.compare("BN") == 0)
		{
			network = "Debug BN";
			debugBN();
		}
		else if (whichTest.compare("SSBits") == 0)
		{
			network = "Debug SS Bits";
			debugSSBits();
		}
		else if (whichTest.compare("SS") == 0)
		{
			network = "Debug SelectShares";
			debugSS();
		}
		else if (whichTest.compare("Maxpool") == 0)
		{
			network = "Debug Maxpool";
			debugMaxpool();
		}
		else if (whichTest.compare("Reduction") == 0)
		{
			network = "Reduction";
			debugReduction();
		}
		else if (whichTest.compare("PartyShare") == 0)
		{
			network = "PartyShare";
			debugPartySS();
		}
		else if (whichTest.compare("PairRandom") == 0)
		{
			network = "PairRandom";
			debugPairRandom();
		}
		else if (whichTest.compare("PosWrap") == 0)
		{
			network = "PosWrap";
			debugPosWrap();
		}
		else if (whichTest.compare("WC-Extension") == 0)
		{
			network = "WC-Extension";
			debugWCExtension();
		}
		else if (whichTest.compare("ZeroRandom"))
		{
			network = "ZeroRandom";
			debugZeroRandom();
		}
		else if (whichTest.compare("Square"))
		{
			network = "Square";
			debugSquare();
		}
		else if (whichTest.compare("Exp"))
		{
			network = "Exp";
			debugExp();
		}
		else
			assert(false && "Unknown debug mode selected");
	}
	else if (str.compare("Test") == 0)
	{
		if (whichTest.compare("Mat-Mul1") == 0)
		{
			network = "Test Mat-Mul1";
			testMatMul(784, 128, 10, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Mat-Mul2") == 0)
		{
			network = "Test Mat-Mul2";
			testMatMul(1, 500, 100, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Mat-Mul3") == 0)
		{
			network = "Test Mat-Mul3";
			testMatMul(1, 100, 1, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLU1") == 0)
		{
			network = "Test ReLU1";
			testRelu(128, 128, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLU2") == 0)
		{
			network = "Test ReLU2";
			testRelu(576, 20, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLU3") == 0)
		{
			network = "Test ReLU3";
			testRelu(64, 16, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLUPrime1") == 0)
		{
			network = "Test ReLUPrime1";
			testReluPrime(128, 128, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLUPrime2") == 0)
		{
			network = "Test ReLUPrime2";
			testReluPrime(576, 20, NUM_ITERATIONS);
		}
		else if (whichTest.compare("ReLUPrime3") == 0)
		{
			network = "Test ReLUPrime3";
			testReluPrime(64, 16, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Conv1") == 0)
		{
			network = "Test Conv1";
			testConvolution(28, 28, 1, 20, 5, 1, 0, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Conv2") == 0)
		{
			network = "Test Conv2";
			testConvolution(28, 28, 1, 20, 3, 1, 0, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Conv3") == 0)
		{
			network = "Test Conv3";
			testConvolution(8, 8, 16, 50, 5, 1, 0, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Maxpool1") == 0)
		{
			network = "Test Maxpool1";
			testMaxpool(24, 24, 20, 2, 2, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Maxpool2") == 0)
		{
			network = "Test Maxpool2";
			testMaxpool(24, 24, 16, 2, 2, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Maxpool3") == 0)
		{
			network = "Test Maxpool3";
			testMaxpool(8, 8, 50, 4, 4, MINI_BATCH_SIZE, NUM_ITERATIONS);
		}
		else if (whichTest.compare("Reduction") == 0)
		{
			network = "reduction";
			testReduction(4);
		}
		else
			assert(false && "Unknown test mode selected");
	}
	else
		assert(false && "Only Debug or Test mode supported");
}
