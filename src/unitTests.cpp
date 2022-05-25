#include "unitTests.h"

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

		funcPartySS<RSSVectorHighType, highBit>(test_result1, a1, size, i);
		funcPartySS<RSSVectorLowType, lowBit>(test_result2, a2, size, i);
		// if (i == partyNum)
		// {
		// 	funcShareSender<RSSVectorHighType, highBit>(test_result1, a1, size);
		// }
		// else
		// {
		// 	funcShareReceiver<RSSVectorHighType, highBit>(test_result1, size, i);
		// }

		// if (i == partyNum)
		// {
		// 	funcShareSender<RSSVectorLowType, lowBit>(test_result2, a2, size);
		// }
		// else
		// {
		// 	funcShareReceiver<RSSVectorLowType, lowBit>(test_result2, size, i);
		// }

#if (!LOG_DEBUG)
		funcReconstruct<RSSVectorHighType, highBit>(test_result1, result_plain1, size, "high output", false);
		funcReconstruct<RSSVectorLowType, lowBit>(test_result2, result_plain2, size, "low output", false);
		// printVector<highBit>(result_plain1, "1", size);
		// printVector<highBit>(result_plain1, "1", size);
		if (partyNum == i)
		{
			assert(equalRssVector(result_plain1, a1, size));
			assert(equalRssVector(result_plain2, a2, size));
		}
#endif
	}
}

void debugPairRandom()
{
	size_t size = 5;
	RSSVectorHighType test_result1(size);
	PrecomputeObject->getPairRand<RSSVectorHighType, highBit>(test_result1, size);

	printRssVector<RSSVectorHighType>(test_result1, "pair random", size);
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

template <typename T>
void checkRight(const vector<T> &a1)
{
	size_t size = a1.size();
	// check right
	if (partyNum == PARTY_A)
	{
		thread *threads = new thread[2];
		vector<T> a2(size);
		vector<T> a3(size);
		threads[0] = thread(receiveVector<T>, ref(a2), nextParty(partyNum), size);
		threads[1] = thread(receiveVector<T>, ref(a3), prevParty(partyNum), size);

		for (int i = 0; i < 2; i++)
			threads[i].join();

		for (size_t i = 0; i < size; i++)
		{
			// cout << a1[i] << " " << a2[i] << " " << a3[i] << endl;
			assert((a1[i] + a2[i] + a3[i]) == 0);
		}
	}
	else
	{
		thread sendthread(sendVector<T>, ref(a1), PARTY_A, size);
		sendthread.join();
	}
}

void debugZeroRandom()
{
	size_t size = 4;
	vector<highBit> a1(size);
	PrecomputeObject->getZeroShareRand<RSSVectorHighType, highBit>(a1, size);
	// printVector(a1, "", size);
	checkRight<highBit>(ref(a1));

	vector<lowBit> b1(size);
	PrecomputeObject->getZeroShareRand<RSSVectorLowType, lowBit>(b1, size);
	// printVector(b1, "", size);
	checkRight<lowBit>(ref(b1));
}

void debugBoolAnd()
{
	int checkParty = PARTY_B;
	size_t size = 5;
	vector<bool> data1(size);
	vector<bool> data2(size);
	for (size_t i = 0; i < size; i++)
	{
		data1[i] = rand() % 2;
		data2[i] = rand() % 2;
	}

	// vector<bool> data1 = {true, false, true, true, false, true, false, false, true, true};	// false, false, false, true, true
	// vector<bool> data2 = {true, false, false, false, true, true, false, true, false, true}; // true, false, true, false, false

	/**
	size_t size = 4;
	vector<bool> data1 = {true, false, true, true};	  // false, false, false, true, true
	vector<bool> data2 = {true, false, false, false}; // true, false, true, false, false
	// and: 1,0,0,0
	**/

	RSSVectorBoolType data_ss1(size);
	RSSVectorBoolType data_ss2(size);
	funcBoolPartySS(data_ss1, data1, size, checkParty);
	funcBoolPartySS(data_ss2, data2, size, checkParty);
	// if (partyNum == checkParty)
	// {
	// 	funcBoolShareSender(data_ss1, data1, size);
	// 	funcBoolShareSender(data_ss2, data2, size);
	// }
	// else
	// {
	// 	funcBoolShareReceiver(data_ss1, checkParty, size);
	// 	funcBoolShareReceiver(data_ss2, checkParty, size);
	// }

	// printBoolRssVec(data_ss1, "data_ss1", size);
	// printBoolRssVec(data_ss2, "data_ss2", size);

	RSSVectorBoolType data_ss(size);
	funcBoolAnd(data_ss, data_ss1, data_ss2, size);

	// vector<bool> data_plain(size);
	// funcBoolRev(data_plain, data_ss, size, "bool plain", true);

	RSSVectorHighType result_h_rss(size);
	RSSVectorLowType result_l_rss(size);
	funcB2A<RSSVectorHighType, highBit>(result_h_rss, data_ss, size, false);
	funcB2A<RSSVectorLowType, lowBit>(result_l_rss, data_ss, size, false);

#if (!LOG_DEBUG)
	// check right

	vector<highBit> plain1(size);
	funcReconstruct<RSSVectorHighType, highBit>(result_h_rss, plain1, size, "high output", false);
	if (partyNum == checkParty)
	{
		for (size_t i = 0; i < size; i++)
		{
			highBit temp1 = data1[i] & data2[i];
			// cout << temp1 << "hh" << plain1[i] << endl;
			assert(plain1[i] == temp1);
		}
	}

	vector<lowBit> plain2(size);
	funcReconstruct<RSSVectorLowType, lowBit>(result_l_rss, plain2, size, "low output", false);

	if (partyNum == checkParty)
	{
		for (size_t i = 0; i < size; i++)
		{
			lowBit temp2 = data1[i] & data2[i];
			// cout << temp2 << "hh" << plain2[i] << endl;
			assert(plain2[i] == temp2);
		}
	}

#endif
}

void debugPosWrap()
{
	size_t size = 3;
	vector<lowBit> data(size);
	for (size_t i = 0; i < size; i++)
	{
		data[i] = (i + 1) << FLOAT_PRECISION;
	}
	// printLowBitVec(data, "data", size);
	// printVector<lowBit>(data, "input", size);

	int checkParty = PARTY_B;

	RSSVectorLowType dataRSS(size);
	RSSVectorHighType wRSS(size);

	funcPartySS<RSSVectorLowType, lowBit>(dataRSS, data, size, checkParty);

	lowBit bias1 = (1l << 30);
	funcAddOneConst(dataRSS, bias1, size);

	// log
	vector<lowBit> input2(size);
	funcReconstruct<RSSVectorLowType, lowBit>(dataRSS, input2, size, "input bias", false);
	printLowBitVec(input2, "input bias", size);

	funcPosWrap(wRSS, dataRSS, size); // test function

	printRssVector<RSSVectorLowType>(dataRSS, "data rss", size);
	vector<highBit> w(size);
	funcReconstruct<RSSVectorHighType, highBit>(wRSS, w, size, "w", false);
	printHighBitVec(w, "w", size);

	if (partyNum == nextParty(checkParty))
	{
		vector<lowBit> extra(size);
		for (size_t i = 0; i < size; i++)
		{
			extra[i] = dataRSS[i].second;
		}

		sendVector<lowBit>(extra, prevParty(partyNum), size);
	}
	else if (partyNum == checkParty)
	{
		// cout << "check" << endl;
		vector<lowBit> extra(size);
		receiveVector<lowBit>(extra, nextParty(partyNum), size);
		for (size_t i = 0; i < size; i++)
		{
			lowBit temp = dataRSS[i].first + dataRSS[i].second;
			highBit result = (temp < dataRSS[i].first || temp < dataRSS[i].second);
			cout << result << " ";
			lowBit temp2 = temp + extra[i];
			result = result + (temp2 < temp || temp2 < extra[i]);
			cout << result << " hhh " << w[i] << endl;
			// assert(result == w[i]);
		}
	}

	// funcPosWrap
}

void debugWCExtension()
{
	size_t size = 3;
	vector<lowBit> data(size);
	for (size_t i = 0; i < size; i++)
	{
		data[i] = (i + 1) << FLOAT_PRECISION;
	}

	printVector<lowBit>(data, "input", size);

	int checkParty = PARTY_B;

	RSSVectorLowType datalow(size);
	funcPartySS<RSSVectorLowType, lowBit>(datalow, data, size, checkParty);
	// if (partyNum == checkParty)
	// {
	// 	funcShareSender<RSSVectorLowType, lowBit>(datalow, data, size);
	// }
	// else
	// {
	// 	funcShareReceiver<RSSVectorLowType, lowBit>(datalow, size, checkParty);
	// }

	printRssVector<RSSVectorLowType>(datalow, "lowbit ss", size);
	funcReconstruct<RSSVectorLowType, lowBit>(datalow, data, size, "low bit plain", true);

	RSSVectorHighType datahigh(size);
	funcWCExtension(datahigh, datalow, size); // test function

	printRssVector<RSSVectorHighType>(datahigh, "highbit ss", size);
	vector<highBit> plain_high(size);
	funcReconstruct<RSSVectorHighType, highBit>(datahigh, plain_high, size, "high bit plain", true);

	for (size_t i = 0; i < size; i++)
	{
		cout << plain_high[i] << " " << data[i] << endl;
		// assert(plain_high[i] == (highBit)data[i]);
	}
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
			// debugPosWrap();
		}
		else if (whichTest.compare("WC-Extension") == 0)
		{
			network = "WC-Extension";
			debugWCExtension();
		}
		else if (whichTest.compare("ZeroRandom") == 0)
		{
			network = "ZeroRandom";
			debugZeroRandom();
		}
		else if (whichTest.compare("BoolAnd") == 0)
		{
			network = "BoolAnd";
			debugBoolAnd();
		}
		else if (whichTest.compare("Square") == 0)
		{
			network = "Square";
			debugSquare();
		}
		else if (whichTest.compare("Exp") == 0)
		{
			network = "Exp";
			debugExp();
		}
		else if (whichTest.compare("Softmax") == 0)
		{
			network = "Softmax";
			debugSoftmax();
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
