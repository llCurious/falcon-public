#include "unitTests.h"

void benchWCExtension()
{
	cout << "wc-extension" << endl;
	size_t dims[4] = {100, 1000, 10000, 100000};
	int cnt = 20;
	uint64_t round = 0;
	uint64_t commsize = 0;

	// commObject.setMeasurement(true);
	// for (int i = 0; i < 4; i++)
	// {
	// 	size_t size = dims[i];
	// 	cout << "dim " << size << endl;
	// 	vector<lowBit> data(size);
	// 	RSSVectorLowType datalow(size);
	// 	RSSVectorHighType datahigh(size);
	// 	funcGetShares(datalow, data);

	// 	round = commObject.getRoundsRecv();
	// 	commsize = commObject.getRecv();

	// 	funcWCExtension(datahigh, datalow, size); // test function

	// 	cout << "round " << commObject.getRoundsRecv() - round << endl;
	// 	// cout << "send round " << commObject.getRoundsSent() << endl;
	// 	cout << "size " << commObject.getRecv() - commsize << endl;
	// 	// cout << "send size" << commObject.getSent() << endl;
	// }
	commObject.setMeasurement(false);
	for (int i = 0; i < 4; i++)
	{
		size_t size = dims[i];
		cout << "dim " << size << endl;
		double time_sum = 0;
		for (int j = 0; j < cnt; ++j)
		{
			vector<lowBit> data(size);
			RSSVectorLowType datalow(size);
			RSSVectorHighType datahigh(size);
			funcGetShares(datalow, data);

			auto start = std::chrono::system_clock::now();
			funcWCExtension(datahigh, datalow, size); // test function
			auto end = std::chrono::system_clock::now();
			double dur = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6);
			time_sum += dur;
			// cout << j << " " << dur << endl;
		}
		cout << time_sum / cnt << endl;
	}
}

void benchMSExtension()
{
	cout << "ms-extension" << endl;
	size_t dims[4] = {100, 1000, 10000, 100000};
	int cnt = 20;
	uint64_t round = 0;
	uint64_t commsize = 0;

	// commObject.setMeasurement(true);
	// for (int i = 0; i < 4; i++)
	// {
	// 	size_t size = dims[i];
	// 	cout << "dim " << size << endl;
	// 	vector<lowBit> data(size);
	// 	RSSVectorLowType datalow(size);
	// 	RSSVectorHighType datahigh(size);
	// 	funcGetShares(datalow, data);

	// 	round = commObject.getRoundsRecv();
	// 	commsize = commObject.getRecv();

	// 	funcMSExtension(datahigh, datalow, size); // test function

	// 	cout << "round " << commObject.getRoundsRecv() - round << endl;
	// 	// cout << "send round " << commObject.getRoundsSent() << endl;
	// 	cout << "size " << commObject.getRecv() - commsize << endl;
	// 	// cout << "send size" << commObject.getSent() << endl;
	// }
	commObject.setMeasurement(false);
	for (int i = 0; i < 4; i++)
	{
		size_t size = dims[i];
		cout << "dim " << size << endl;
		double time_sum = 0;
		for (int j = 0; j < cnt; ++j)
		{
			vector<lowBit> data(size);
			RSSVectorLowType datalow(size);
			RSSVectorHighType datahigh(size);
			funcGetShares(datalow, data);

			auto start = std::chrono::system_clock::now();
			funcMSExtension(datahigh, datalow, size); // test function
			auto end = std::chrono::system_clock::now();
			double dur = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6);
			time_sum += dur;
			// cout << j << " " << dur << endl;
		}
		cout << time_sum / cnt << endl;
	}
}

template <typename Vec>
void runBN(BNLayerOpt *layer, Vec &forward_output, Vec &input_act, Vec &grad, Vec &x_grad)
{
	layer->forward(input_act);
	forward_output = *layer->getActivation();
	// print_vector(forward_output, "FLOAT", "BN Forward", forward_output.size());

	*(layer->getDelta()) = grad;
	layer->computeDelta(x_grad);
	layer->updateEquations(input_act);
}

template <typename Vec>
void runBN(BNLayer *layer, Vec &forward_output, Vec &input_act, Vec &grad, Vec &x_grad)
{
	layer->forward(input_act);
	forward_output = *layer->getActivation();
	print_vector(forward_output, "FLOAT", "BN Forward", forward_output.size());

	*(layer->getDelta()) = grad;
	layer->computeDelta(x_grad);
	layer->updateEquations(input_act);
}

template <typename Vec, typename T>
void getBNInput(vector<float> &x_raw, vector<float> &grad_raw, Vec &input_act, Vec &grad, size_t B, size_t D)
{
	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<Vec, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<Vec, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(x_raw).name() << endl;
	}

	size_t size = B * D;
	for (size_t i = 0; i < B; i++)
	{
		for (size_t j = 0; j < D; j++)
		{
			// x_raw.push_back(rand() % 10);
			// grad_raw.push_back(rand() % 3);
			x_raw[i * D + j] = rand() % 5 + 1;
			grad_raw[i * D + j] = rand() % 2 + 1;
		}
	}

	// FXP representation
	vector<T> x_p(size), grad_p(size);
	for (size_t i = 0; i < size; i++)
	{
		x_p[i] = x_raw[i] * (1 << float_precision);
		grad_p[i] = grad_raw[i] * (1 << float_precision);
	}

	funcGetShares(input_act, x_p);
	funcGetShares(grad, grad_p);
}

/**
 * @brief get bn input from file
 *
 * @param x_raw
 * @param grad_raw
 * @param input_act
 * @param grad
 * @param B
 * @param D
 */
template <typename Vec, typename T>
void getBNInput(string filename, vector<float> &x_raw, vector<float> &grad_raw, Vec &input_act, Vec &grad, size_t B, size_t D)
{
	size_t size = B * D;

	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<Vec, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<Vec, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(x_raw).name() << endl;
	}

	ifstream infile;
	infile.open(filename.data());
	assert(infile.is_open());
	int i = 0, j = 0;
	char buf[1024] = {0};
	while (infile >> buf)
	{
		if (i < B)
		{
			x_raw[i * D + j] = stof(buf);
		}
		else
		{
			grad_raw[(i - B) * D + j] = stof(buf);
		}
		++j;
		if (j == D)
		{
			j = 0;
			++i;
		}
	}
	infile.close();

	// FXP representation
	vector<highBit> x_p(size), grad_p(size);
	for (size_t i = 0; i < size; i++)
	{
		x_p[i] = x_raw[i] * (1 << float_precision);
		grad_p[i] = grad_raw[i] * (1 << float_precision);
		// cout << x_p[i] << " " << grad_p[i] << " ";
	}

	funcGetShares(input_act, x_p);
	funcGetShares(grad, grad_p);
}

template <typename Vec, typename T>
void benchBN()
{
	size_t ds[2] = {100, 1000};
	// batch size: 32,64,128,256
	size_t B = (1 << LOG_MINI_BATCH), D;
	clock_t start, end;
	double time_sum = 0;
	int cnt = 10;

	// string infile = "./scripts/test_data/input.csv";
	// string outfile = "./scripts/test_data/output.csv";

	uint64_t round = 0;
	uint64_t commsize = 0;

	for (size_t j = 0; j < 2; j++)
	{
		D = ds[j];
		size_t size = B * D;

		vector<float> x_raw(size);
		vector<float> grad_raw(size);

		Vec forward_output(size);
		Vec x_grad(size);

		Vec input_act(size), grad(size);
		if (IS_FALCON)
		{
			BNConfig *bn_conf = new BNConfig(D, B);
			BNLayer *layer = new BNLayer(bn_conf, 0);

			getBNInput<Vec, T>(x_raw, grad_raw, input_act, grad, B, D);

			// comm test
			round = commObject.getRoundsRecv();
			commsize = commObject.getRecv();
			runBN<Vec>(layer, forward_output, input_act, grad, x_grad);
			cout << "round: " << commObject.getRoundsRecv() - round << "  size: " << commObject.getRecv() - commsize << endl;

			for (size_t i = 0; i < cnt; i++)
			{
				getBNInput<Vec, T>(x_raw, grad_raw, input_act, grad, B, D);

				// time test
				start = clock();
				runBN<Vec>(layer, forward_output, input_act, grad, x_grad);

				end = clock();
				double dur = (double)(end - start) / CLOCKS_PER_SEC;
				time_sum += dur;
			}
		}
		else
		{
			BNConfig *bn_conf = new BNConfig(D, B);
			BNLayerOpt *layer = new BNLayerOpt(bn_conf, 0);

			getBNInput<Vec, T>(x_raw, grad_raw, input_act, grad, B, D);

			// comm test
			round = commObject.getRoundsRecv();
			commsize = commObject.getRecv();
			runBN<Vec>(layer, forward_output, input_act, grad, x_grad);
			cout << "round: " << commObject.getRoundsRecv() - round << "  size: " << commObject.getRecv() - commsize << endl;

			for (size_t i = 0; i < cnt; i++)
			{
				getBNInput<Vec, T>(x_raw, grad_raw, input_act, grad, B, D);

				// time test
				start = clock();
				runBN<Vec>(layer, forward_output, input_act, grad, x_grad);

				end = clock();
				double dur = (double)(end - start) / CLOCKS_PER_SEC;
				time_sum += dur;
			}
		}
		cout << B << " " << D << " " << time_sum / cnt << endl;
	}
}

template <typename Vec, typename T>
void benchBNAcc()
{
	// batch size: 32,64,128,256
	size_t B = (1 << LOG_MINI_BATCH), D = 100;
	clock_t start, end;
	double time_sum = 0;

	string infile = "./scripts/test_data/input_bn.csv";
	string outfile = "./scripts/test_data/output_bn_mix.csv";

	uint64_t round = 0;
	uint64_t commsize = 0;

	size_t size = B * D;

	vector<float> x_raw(size);
	vector<float> grad_raw(size);

	Vec input_act(size), grad(size);

	Vec forward_output(size);
	Vec x_grad(size);
	Vec gammagrad(D), betagrad(D);

	if (IS_FALCON)
	{
		BNConfig *bn_conf = new BNConfig(D, B);
		BNLayer *layer = new BNLayer(bn_conf, 0);

		getBNInput<Vec, T>(infile, x_raw, grad_raw, input_act, grad, B, D);
		runBN<Vec>(layer, forward_output, input_act, grad, x_grad);

		gammagrad = *layer->getGammaGrad();
		betagrad = *layer->getBetaGrad();
	}
	else
	{
		BNConfig *bn_conf = new BNConfig(D, B);
		BNLayerOpt *layer = new BNLayerOpt(bn_conf, 0);

		getBNInput<Vec, T>(infile, x_raw, grad_raw, input_act, grad, B, D);
		runBN<Vec>(layer, forward_output, input_act, grad, x_grad);

		gammagrad = *layer->getGammaGrad();
		betagrad = *layer->getBetaGrad();
	}

	// record output, x_forward, x_grad, gamma_grad, beta_grad
	vector<T> x_f(size), x_g(size), gamma_g(D), beta_g(D);
	funcReconstruct(forward_output, x_f, size, "x_forward", false);
	funcReconstruct(x_grad, x_g, size, "x_grad", false);
	funcReconstruct(gammagrad, gamma_g, D, "gamma_grad", false);
	funcReconstruct(betagrad, beta_g, D, "beta_grad", false);

	mat2file<T>(x_f, x_g, gamma_g, beta_g, outfile, B, D);
}

void benchSoftMax()
{
	// d=10/200，batch=100/1000/10000/100000
	size_t ds[2] = {10, 200};
	size_t batchs[3] = {100, 1000, 10000};
	// size_t ds[1] = {200};
	// size_t batchs[1] = {10000};
	size_t cnt = 4;

	size_t size;

	uint64_t round = 0;
	uint64_t commsize = 0;

	clock_t start, end;

	for (int d : ds)
	{
		for (int batch : batchs)
		{
			size = d * batch;

			RSSVectorHighType a(size), b(size);

			// comm
			round = commObject.getRoundsRecv();
			commsize = commObject.getRecv();

			funcSoftmax(a, b, batch, d, false);

			cout << "round: " << commObject.getRoundsRecv() - round << "  size: " << commObject.getRecv() - commsize << endl;

			// time
			double time_sum = 0;
			for (size_t i = 0; i < cnt; i++)
			{
				start = clock();
				funcSoftmax(a, b, batch, d, false);

				end = clock();
				double dur = (double)(end - start) / CLOCKS_PER_SEC;
				cout << dur << endl;
				time_sum += dur;
			}
			cout << batch << " " << d << " " << time_sum / cnt << endl;
		}
	}
}

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
	size_t size = 100;
	vector<highBit> a(size);

	for (size_t i = 0; i < size; i++)
	{
		lowBit temp = rand();
		a[i] = temp;
	}

	RSSVectorHighType test_data(size);
	RSSVectorLowType test_result(size);

	funcGetShares(test_data, a);

	funcReduction(test_result, test_data, size);

	// cout << "begin print ss" << endl;

	// print_vector(test_data, "UNSIGNED", "test_data ss", test_data.size());
	// print_vector(test_result, "UNSIGNED", "test_result ss", test_result.size());

	vector<highBit> data_plain(size);
	vector<lowBit> result_plain(size);

	// cout << "Reconstruct" << endl;

#if (!LOG_DEBUG)
	funcReconstruct<RSSVectorHighType, highBit>(test_data, data_plain, size, "input", false);
	funcReconstruct<RSSVectorLowType, lowBit>(test_result, result_plain, size, "output", false);

	for (size_t i = 0; i < size; i++)
	{
		// cout << data_plain[i] << " " << result_plain[i] << endl;
		assert((lowBit)(data_plain[i]) == (lowBit)(result_plain[i]));
	}
#endif
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
	size_t size = 100;
	vector<lowBit> data(size);
	size_t i = 0;
	data[i] = -(1 << 30);
	// cout << bitset<32>(data[i]) << endl;
	i++;
	for (; i < size / 2; i++)
	{
		data[i] = data[i - 1] + 1;
	}
	for (; i < size; i++)
	{
		data[i] = ((i) << FLOAT_PRECISION);
	}
	// printVector<lowBit>(data, "input", size);

	int checkParty = PARTY_B;

	RSSVectorLowType dataRSS(size);
	RSSVectorHighType wRSS(size);

	funcPartySS<RSSVectorLowType, lowBit>(dataRSS, data, size, checkParty);

	lowBit bias1 = (1l << 30);
	funcAddOneConst(dataRSS, bias1, size);

	// log
	// vector<lowBit> input2(size);
	// funcReconstruct<RSSVectorLowType, lowBit>(dataRSS, input2, size, "input bias", false);
	// printLowBitVec(input2, "input bias", size);

	funcPosWrap(wRSS, dataRSS, size); // test function

	// printRssVector<RSSVectorLowType>(dataRSS, "data rss", size);
	vector<highBit> w(size);
	funcReconstruct<RSSVectorHighType, highBit>(wRSS, w, size, "w", false);
	// printHighBitVec(w, "w", size);

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

		// printRSSLowBitVec(dataRSS, "data rss", size);
		// printLowBitVec(extra, "extra", size);
		for (size_t i = 0; i < size; i++)
		{
			lowBit temp = dataRSS[i].first + dataRSS[i].second;
			highBit result = (temp < dataRSS[i].first || temp < dataRSS[i].second);
			// cout << result << " ";
			lowBit temp2 = temp + extra[i];
			result = result + (temp2 < temp || temp2 < extra[i]);
			// cout << result << " hhh " << w[i] << endl;
			assert(result == w[i]);
		}
	}

	// funcPosWrap
}

void debugWCExtension()
{
	// cout << "Debug WC-Share Extension" << endl;
	size_t size = 10240;
	vector<lowBit> data(size);
	size_t i = 0;
	data[i] = -(1 << 30);
	// cout << bitset<32>(data[i]) << endl;
	i++;
	for (; i < size / 2; i++)
	{
		data[i] = data[i - 1] + 1;
	}
	data[i] = (1 << 30) - 1;
	i++;
	for (; i < size; i++)
	{
		data[i] = data[i - 1] - 1;
	}
	// printLowBitVec(data, "input", size);
	// printVector<lowBit>(data, "input", size);

	int checkParty = PARTY_B;

	RSSVectorLowType datalow(size);
	funcPartySS<RSSVectorLowType, lowBit>(datalow, data, size, checkParty);

	RSSVectorHighType datahigh(size);
	funcWCExtension(datahigh, datalow, size); // test function

	// printRssVector<RSSVectorHighType>(datahigh, "highbit ss", size);
	vector<highBit> plain_high(size);
	funcReconstruct<RSSVectorHighType, highBit>(datahigh, plain_high, size, "high bit plain", false);
	// printHighBitVec(plain_high, "", size);

	for (size_t i = 0; i < size; i++)
	{
		// cout << (int)plain_high[i] << " " << (int)data[i] << endl;
		assert((int)plain_high[i] == (int)(highBit)data[i]);
	}
}

void debugMixedShareGen()
{
	size_t size = 1000;
	RSSVectorHighType an(size);
	RSSVectorLowType am(size);
	RSSVectorHighType rmsb(size);
	funcMixedShareGen(an, am, rmsb, size);

	vector<highBit> high_plain(size);
	vector<lowBit> low_plain(size);
	vector<highBit> msb_plain(size);
	funcReconstruct<RSSVectorHighType, highBit>(an, high_plain, size, "high plain", false);
	funcReconstruct<RSSVectorLowType, lowBit>(am, low_plain, size, "low plain", false);
	funcReconstruct<RSSVectorHighType, highBit>(rmsb, msb_plain, size, "msb plain", false);
	for (size_t i = 0; i < size; i++)
	{
		// cout << (int)high_plain[i] << " " << (int)low_plain[i] << endl;
		int temp = int(low_plain[i]);
		highBit msb = temp < 0 ? 1 : 0;
		assert(msb == msb_plain[i]);
		// cout << msb << " " << msb_plain[i] << endl;
		assert((lowBit)high_plain[i] == low_plain[i]);
	}
}

void debugMSExtension()
{
	size_t size = 30;
	vector<float> data_row = {1.59265, 1.59265, 1.59265, 1.59265, 1.59265, 1.59265, 1.59265, 1.59265, 1.59265, 1.59265, 5.35291, 5.35291, 5.35291, 5.35291, 5.35291, 5.35291, 5.35291, 5.35291, 5.35291, 5.35291, 6.7373, 6.7373, 6.7373, 6.7373, 6.7373, 6.7373, 6.7373, 6.7373, 6.7373, 6.7373};
	vector<lowBit> data(size);
	for (size_t i = 0; i < size; i++)
	{
		data[i] = data_row[i] * (1l << LOW_PRECISION);
	}

	// size_t i = 0;
	// data[i] = -(1 << 30);
	// // cout << bitset<32>(data[i]) << endl;
	// i++;
	// for (; i < size / 2; i++)
	// {
	// 	data[i] = data[i - 1] + 1;
	// }
	// data[i] = (1 << 30) - 1;
	// i++;
	// for (; i < size; i++)
	// {
	// 	data[i] = data[i - 1] - 1;
	// }
	// printLowBitVec(data, "input", size);
	// printVector<lowBit>(data, "input", size);
	printVectorReal<lowBit>(data, "input", size);

	int checkParty = PARTY_B;

	RSSVectorLowType datalow(size);
	funcPartySS<RSSVectorLowType, lowBit>(datalow, data, size, checkParty);

	RSSVectorHighType datahigh(size);
	funcMSExtension(datahigh, datalow, size); // test function

	// printRssVector<RSSVectorHighType>(datahigh, "highbit ss", size);
	vector<highBit> plain_high(size);
	funcReconstruct<RSSVectorHighType, highBit>(datahigh, plain_high, size, "high bit plain", false);
	// printVector<highBit>(plain_high, "", size);
	printVectorReal<highBit>(plain_high, "output", size);

	for (size_t i = 0; i < size; i++)
	{
		// cout << (int)plain_high[i] << " " << (int)data[i] << endl;
		assert((int)plain_high[i] == (int)data[i]);
	}
}

void debugTruncAndReduce()
{
	int checkParty = PARTY_B;

	int trunc_bits = 10;
	size_t size = 500;
	vector<highBit> data(size);

	RSSVectorHighType input(size);
	RSSVectorLowType output(size);

	// test data
	int i;
	for (i = 0; i < size / 2; ++i)
	{
		int temp = -random();
		data[i] = temp;
	}
	for (; i < size; ++i)
	{
		int temp = random();
		data[i] = temp;
	}
	funcPartySS<RSSVectorHighType, highBit>(input, data, size, checkParty);

	funcTruncAndReduce(output, input, trunc_bits, size);

	// check
	vector<lowBit> output_p(size);
	funcReconstruct<RSSVectorLowType, lowBit>(output, output_p, size, "output plain", false);

	if (partyNum == checkParty)
	{
		for (size_t i = 0; i < size; i++)
		{
			int t1 = (int(data[i]) >> trunc_bits);
			int t2 = int(output_p[i]);
			assert(t2 == t1 || t2 == t1 - 1 || t2 == t1 - 2);
			// cout << (int(data[i]) >> trunc_bits) << " " << int(output_p[i]) << endl;
		}
	}
}

void debugBNLayer()
{
	cout << "Debug Batch Normalization Layer" << endl;
	size_t B = 4, D = 5;
	size_t size = B * D;

	// Floating point representation
	vector<float> x_raw = {
		1, 2, 3, 4, 5,
		1, 3, 5, 7, 8,
		1, 2, 3, 6, 6,
		1, 2, 4, 5, 6};

	// vector<float> grad_raw = {
	// 	1, 1, 1, 1, 1,
	// 	1, 1, 1, 1, 1,
	// 	1, 1, 1, 1, 1,
	// 	1, 1, 1, 1, 1};
	vector<float> grad_raw = {
		1, 1, 1, 1, 1,
		1, 2, 1, 1, 1,
		2, 2, 1, 1, 2,
		1, 2, 2, 1, 2};

	// FXP representation
	vector<highBit> x_p(size), grad_p(size);
	for (size_t i = 0; i < size; i++)
	{
		x_p[i] = x_raw[i] * (1 << FLOAT_PRECISION);
		grad_p[i] = grad_raw[i] * (1 << FLOAT_PRECISION);
	}

	// Public to secret
	RSSVectorHighType input_act(size), grad(size);
	funcGetShares(input_act, x_p);
	funcGetShares(grad, grad_p);

	BNConfig *bn_conf = new BNConfig(D, B);
	BNLayerOpt *layer = new BNLayerOpt(bn_conf, 0);
	layer->printLayer();

	// Forward.
	RSSVectorHighType forward_output(size), backward_output(size);
	// layer->forward(input_act);
	// forward_output = *layer->getActivation();
	// print_vector(forward_output, "FLOAT", "BN Forward", size);

	// // Backward.
	// RSSVectorHighType x_grad(size);
	// // layer->backward(grad);
	// *(layer->getDelta()) = grad;
	// layer->computeDelta(x_grad);
	// print_vector(x_grad, "FLOAT", "BN Backward- X", size);

	// // Noted: i recommend print the calculated delta for beta and gamma in BNLayerOpt.
	// layer->updateEquations(input_act);
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
		else if (whichTest.compare("WCExtension") == 0)
		{
			network = "WCExtension";
			debugWCExtension();
		}
		else if (whichTest.compare("MSExtension") == 0)
		{
			network = "MSExtension";
			debugMSExtension();
		}
		else if (whichTest.compare("ProbTruncation") == 0)
		{
			network = "ProbTruncation";
			debugProbTruncation<RSSVectorHighType, highBit>();
			debugProbTruncation<RSSVectorLowType, lowBit>();
		}
		else if (whichTest.compare("TruncAndReduce") == 0)
		{
			network = "TruncAndReduce";
			debugTruncAndReduce();
		}
		else if (whichTest.compare("RandBit") == 0)
		{
			network = "RandBit";
			debugRandBit<RSSVectorLongType, longBit, RSSVectorHighType>();
		}
		else if (whichTest.compare("Reciprocal") == 0)
		{
			network = "Reciprocal";
			debugReciprocal<RSSVectorHighType, highBit>();
		}
		else if (whichTest.compare("funcDivisionByNR") == 0)
		{
			network = "funcDivisionByNR";
			// debugDivisionByNR<RSSVectorHighType, highBit>();
			debugDivisionByNR<RSSVectorLowType, lowBit>();
		}
		else if (whichTest.compare("InverseSqrt") == 0)
		{
			network = "InverseSqrt";
			debugInverseSqrt<RSSVectorHighType, highBit>();
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
		else if (whichTest.compare("MixedShareGen") == 0)
		{
			network = "MixedShareGen";
			debugMixedShareGen();
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
			debugSoftmax<RSSVectorHighType, highBit>();
			// debugSoftmax<RSSVectorHighType, highBit>();
		}
		else if (whichTest.compare("BNLayer") == 0)
		{
			network = "BNLayer";
			debugBNLayer<RSSVectorMyType, myType>();
			/// debugBNLayer<RSSVectorLowType, lowBit>();
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
	else if (str.compare("Bench") == 0)
	{
		if (whichTest.compare("MSExtension") == 0)
		{
			network = "MSExtension";
			benchMSExtension();
		}
		else if (whichTest.compare("WCExtension") == 0)
		{
			network = "WCExtension";
			benchWCExtension();
		}
		else if (whichTest.compare("BN") == 0)
		{
			network = "BN";
			benchBN<RSSVectorMyType, myType>();
			// benchBNAcc<RSSVectorMyType, myType>();
		}
		else if (whichTest.compare("SoftMax") == 0)
		{
			network = "SoftMax";
			// benchSoftMaxAcc<RSSVectorHighType, highBit>();
			benchSoftMaxAcc<RSSVectorLowType, lowBit>();
		}
	}
	else
		assert(false && "Only Debug or Test mode supported");
}
