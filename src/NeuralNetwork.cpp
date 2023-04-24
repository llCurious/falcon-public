
#pragma once
#include "tools.h"
#include "FCLayer.h"
#include "CNNLayer.h"
#include "MaxpoolLayer.h"
#include "ReLULayer.h"
#include "BNLayer.h"
#include "BNLayerOpt.h"
#include "NeuralNetwork.h"
#include "Functionalities.h"
using namespace std;

extern size_t INPUT_SIZE;
extern size_t LAST_LAYER_SIZE;
extern bool WITH_NORMALIZATION;
extern bool LARGE_NETWORK;

NeuralNetwork::NeuralNetwork(NeuralNetConfig *config)
	: inputData(INPUT_SIZE * MINI_BATCH_SIZE),
	  low_inputData(INPUT_SIZE * MINI_BATCH_SIZE),
	  outputData(LAST_LAYER_SIZE * MINI_BATCH_SIZE),
	  softmax_output(LAST_LAYER_SIZE * MINI_BATCH_SIZE)
{
	for (size_t i = 0; i < NUM_LAYERS; ++i)
	{
		if (config->layerConf[i]->type.compare("FC") == 0)
		{
			FCConfig *cfg = static_cast<FCConfig *>(config->layerConf[i]);
			layers.push_back(new FCLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("CNN") == 0)
		{
			CNNConfig *cfg = static_cast<CNNConfig *>(config->layerConf[i]);
			layers.push_back(new CNNLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("Maxpool") == 0)
		{
			MaxpoolConfig *cfg = static_cast<MaxpoolConfig *>(config->layerConf[i]);
			layers.push_back(new MaxpoolLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("ReLU") == 0)
		{
			ReLUConfig *cfg = static_cast<ReLUConfig *>(config->layerConf[i]);
			layers.push_back(new ReLULayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("BN") == 0)
		{
			BNConfig *cfg = static_cast<BNConfig *>(config->layerConf[i]);
			layers.push_back(new BNLayerOpt(cfg, i));
			// layers.push_back(new BNLayer(cfg, i));
		}
		else
			error("Only FC, CNN, ReLU, Maxpool, and BN layer types currently supported");
	}
}

NeuralNetwork::~NeuralNetwork()
{
	for (vector<Layer *>::iterator it = layers.begin(); it != layers.end(); ++it)
		delete (*it);

	layers.clear();
}

void NeuralNetwork::forward()
{
	log_print("NN.forward");

	layers[0]->forward(low_inputData);
	if (LARGE_NETWORK)
		cout << "Forward \t" << layers[0]->layerNum << " completed..." << endl;

	// cout << "----------------------------------------------" << endl;
	// cout << "DEBUG: forward() at NeuralNetwork.cpp" << endl;
	// print_vector(inputData, "FLOAT", "inputData:", 784);
	// print_vector(*((CNNLayer*)layers[0])->getWeights(), "FLOAT", "w0:", 20);
	// print_vector((*layers[0]->getActivation()), "FLOAT", "a0:", 1000);

	for (size_t i = 1; i < NUM_LAYERS; ++i)
	{
		cout << "Layer" << i << endl;
		layers[i]->forward(*(layers[i - 1]->getActivation()));
		if (LARGE_NETWORK)
			cout << "Forward \t" << layers[i]->layerNum << " completed..." << endl;

		// print_vector((*layers[i]->getActivation()), "FLOAT", "Activation Layer"+to_string(i),
		// 			(*layers[i]->getActivation()).size());
		// print_vector((*layers[i]->getActivation()), "FLOAT", "Activation Layer "+to_string(i), 100);
	}
	// print_vector(inputData, "FLOAT", "Input:", 784);
	// cout << "size of output: " << (*layers[NUM_LAYERS-1]->getActivation()).size() << endl;
	// print_vector((*layers[NUM_LAYERS-1]->getActivation()), "FLOAT", "Output:", 10);
}

void NeuralNetwork::backward()
{
	log_print("NN.backward");
	computeDelta();
	cout << "----------------------------------" << endl;
	cout << "computeDelta Done" << endl;
	cout << "----------------------------------" << endl;
	updateEquations();
}

void NeuralNetwork::computeDelta()
{
	log_print("NN.computeDelta");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	size_t size = rows * columns;
	size_t index;

	if (WITH_NORMALIZATION)
	{
		BackwardVectorType rowSum(size, make_pair(0, 0));
		BackwardVectorType quotient(size, make_pair(0, 0));

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				rowSum[i * columns] = rowSum[i * columns] +
									  (*(layers[NUM_LAYERS - 1]->getHighActivation()))[i * columns + j];

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				rowSum[i * columns + j] = rowSum[i * columns];
		if (IS_FALCON)
		{
			funcDivision(*(layers[NUM_LAYERS - 1]->getHighActivation()), rowSum, quotient, size);
		}
		else
		{
			funcDivisionByNR(*(layers[NUM_LAYERS - 1]->getHighActivation()), rowSum, quotient, size);
		}
		// funcDivision(*(layers[NUM_LAYERS-1]->getActivation()), rowSum, quotient, size);

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				index = i * columns + j;
				(*(layers[NUM_LAYERS - 1]->getDelta()))[index] = quotient[index] - outputData[index];
			}
	}
	else
	{
		/**
		 * Updated Softmax + CE gradients computation
		 */
		if (USE_SOFTMAX_CE)
		{
			funcSoftmax(*(layers[NUM_LAYERS - 1]->getHighActivation()), softmax_output, rows, columns, false);
			subtractVectors(softmax_output, outputData, *(layers[NUM_LAYERS - 1]->getDelta()), size);
			// funcProbTruncation<BackwardVectorType, BackwardType>(*(layers[NUM_LAYERS - 1]->getDelta()), LOG_MINI_BATCH, size);
			// print_vector(*(layers[NUM_LAYERS - 1]->getHighActivation()), "FLOAT", "predict", size);
			// print_vector(softmax_output, "FLOAT", "predict_softmax", size);
			// print_vector(*(layers[NUM_LAYERS - 1]->getDelta()), "FLOAT", "loss", (layers[NUM_LAYERS - 1]->getDelta())->size());
			// print_vector(outputData, "FLOAT", "target", LAST_LAYER_SIZE * MINI_BATCH_SIZE);
		}
		else
		{
			/**
			 * Updated MSE
			 * **/
			BackwardVectorType diff(size);
			subtractVectors(*(layers[NUM_LAYERS - 1]->getHighActivation()), outputData, diff, size);
			*(layers[NUM_LAYERS - 1]->getDelta()) = diff;
			// print_vector(*(layers[NUM_LAYERS - 1]->getHighActivation()), "FLOAT", "predict", 100);
			// print_vector(outputData, "FLOAT", "label", 10);
			// print_vector(diff, "FLOAT", "diff", diff.size());
			// funcTruncate(diff, LOG_MINI_BATCH, size);
		}

		/**
		 * Raw implementation
		 * **/
		// for (size_t i = 0; i < rows; ++i)
		// 	for (size_t j = 0; j < columns; ++j)
		// 	{
		// 		index = i * columns + j;
		// 		(*(layers[NUM_LAYERS-1]->getDelta()))[index] =
		// 		(*(layers[NUM_LAYERS-1]->getActivation()))[index] - outputData[index];
		// 	}
	}

	if (LARGE_NETWORK)
		cout << "Delta last layer completed." << endl;

	for (size_t i = NUM_LAYERS - 1; i > 0; --i)
	{
		// cout << "Delta " << i << endl;
		layers[i]->computeDelta(*(layers[i - 1]->getDelta()));
		if (LARGE_NETWORK)
			cout << "Delta \t\t" << layers[i]->layerNum << " completed..." << endl;
	}
}

void NeuralNetwork::updateEquations()
{
	log_print("NN.updateEquations");

	for (size_t i = NUM_LAYERS - 1; i > 0; --i)
	{
		// cout << "Update " << i << endl;
		layers[i]->updateEquations(*(layers[i - 1]->getHighActivation()));
		if (LARGE_NETWORK)
			cout << "Update Eq. \t" << layers[i]->layerNum << " completed..." << endl;
	}

	layers[0]->updateEquations(inputData);
	if (LARGE_NETWORK)
		cout << "First layer update Eq. completed." << endl;
}

void NeuralNetwork::predict(RSSVectorMyType &maxIndex)
{
	log_print("NN.predict");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	BackwardVectorType max(rows);
	RSSVectorSmallType maxPrime(rows * columns);

	funcMaxpool(*(layers[NUM_LAYERS - 1]->getHighActivation()), max, maxPrime, rows, columns);
}

/* new implementation, may still have bug and security flaws */
float NeuralNetwork::getAccuracy()
{
	vector<size_t> counter(2);
	log_print("NN.getAccuracy");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;

	BackwardVectorType max(rows);
	RSSVectorSmallType maxPrime(rows * columns);
	BackwardVectorType temp_max(rows), temp_groundTruth(rows);
	RSSVectorSmallType temp_maxPrime(rows * columns);

	vector<BackwardType> groundTruth(rows * columns);
	vector<smallType> prediction(rows * columns);

	// reconstruct ground truth from output data
	funcReconstruct(outputData, groundTruth, rows * columns, "groundTruth", false);
	// print_vector(outputData, "FLOAT", "outputData:", rows*columns);

	// reconstruct prediction from neural network
	funcMaxpool((*(layers[NUM_LAYERS - 1])->getHighActivation()), temp_max, temp_maxPrime, rows, columns);
	funcReconstructBit(temp_maxPrime, prediction, rows * columns, "prediction", false);

	for (int i = 0, index = 0; i < rows; ++i)
	{
		counter[1]++;
		for (int j = 0; j < columns; j++)
		{
			index = i * columns + j;
			if ((int)groundTruth[index] * (int)prediction[index] ||
				(!(int)groundTruth[index] && !(int)prediction[index]))
			{
				if (j == columns - 1)
				{
					counter[0]++;
				}
			}
			else
			{
				break;
			}
		}
	}

	// for (myType target_label: groundTruth) cout << target_label << ", ";
	// cout << endl;
	// for (smallType predict_label: prediction) {
	// 	cout << unsigned(predict_label) << ", ";
	// }
	// cout << endl;

	cout << "Rolling accuracy: " << counter[0] << " out of " 
		 << counter[1] << " (" << (counter[0]*100.0/counter[1]) << " %)" << endl;
	return (counter[0]*100.0/counter[1]);
}

float NeuralNetwork::getCorrectCount()
{
	vector<size_t> counter(2);
	log_print("NN.getCorrectCount");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;

	BackwardVectorType max(rows);
	RSSVectorSmallType maxPrime(rows * columns);
	BackwardVectorType temp_max(rows), temp_groundTruth(rows);
	RSSVectorSmallType temp_maxPrime(rows * columns);

	vector<BackwardType> groundTruth(rows * columns);
	vector<smallType> prediction(rows * columns);

	// reconstruct ground truth from output data
	funcReconstruct(outputData, groundTruth, rows * columns, "groundTruth", false);
	// print_vector(outputData, "FLOAT", "outputData:", rows*columns);

	// reconstruct prediction from neural network
	funcMaxpool((*(layers[NUM_LAYERS - 1])->getHighActivation()), temp_max, temp_maxPrime, rows, columns);
	funcReconstructBit(temp_maxPrime, prediction, rows * columns, "prediction", false);

	for (int i = 0, index = 0; i < rows; ++i)
	{
		counter[1]++;
		for (int j = 0; j < columns; j++)
		{
			index = i * columns + j;
			if ((int)groundTruth[index] * (int)prediction[index] ||
				(!(int)groundTruth[index] && !(int)prediction[index]))
			{
				if (j == columns - 1)
				{
					counter[0]++;
				}
			}
			else
			{
				break;
			}
		}
	}

	return counter[0];
}

float NeuralNetwork::getLoss() {
	log_print("NN.getLoss");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	size_t size = rows * columns;
	size_t index = 0;

	// RSSVectorMyType diff(size), square(size);
	// subtractVectors(*(layers[NUM_LAYERS-1]->getActivation()), outputData, diff, size);
	// funcSquare(diff, square, size);

	// print_vector(diff, "FLOAT", "loss_diff", size);

	// RSSVectorMyType loss_sum(1);
	// for (index = 0; index < size; index++) {
	// 	loss_sum[0] = loss_sum[0] + square[index];
	// }

	// print_vector(loss_sum, "FLOAT", "loss", 1);

	// Plain-text version
	vector<BackwardType> reconst_label(size);
	vector<float> reconst_label_float(size);
	funcReconstruct(outputData, reconst_label, size, "NN label", false);
	for (size_t i = 0; i < size; i++)
	{
		if (sizeof(BackwardType) == 4)
		{ // int32
			reconst_label_float[i] = (static_cast<int32_t>(reconst_label[i])) / (float)(1 << FORWARD_PRECISION);
		}
		else if (sizeof(BackwardType) == 8)
		{ // int64
			reconst_label_float[i] = (static_cast<int64_t>(reconst_label[i])) / (float)(1 << BACKWARD_PRECISION);
		}
	}

	float loss = 0;

	if (USE_SOFTMAX_CE)
	{ // Cross Entropy
		vector<BackwardType> reconst_y_soft(size);
		vector<float> reconst_y_soft_float(size);
		funcReconstruct(softmax_output, reconst_y_soft, size, "NN output softmax", false);
		for (size_t i = 0; i < size; i++)
		{
			if (sizeof(BackwardType) == 4)
			{ // int32
				reconst_y_soft_float[i] = (static_cast<int32_t>(reconst_y_soft[i])) / (float)(1 << FORWARD_PRECISION);
			}
			else if (sizeof(BackwardType) == 8)
			{ // int64
				reconst_y_soft_float[i] = (static_cast<int64_t>(reconst_y_soft[i])) / (float)(1 << BACKWARD_PRECISION);
			}
			// avoid log(0) cause nan
			reconst_y_soft_float[i] = reconst_y_soft_float[i] == 0 ? 1e-6 : reconst_y_soft_float[i];
			loss += -(reconst_label_float[i] * log(reconst_y_soft_float[i]));
		}
	}
	else
	{ // MSE
		vector<BackwardType> reconst_y(size);
		vector<float> reconst_y_float(size);
		funcReconstruct(*(layers[NUM_LAYERS - 1]->getHighActivation()), reconst_y, size, "NN output", false);
		for (size_t i = 0; i < size; i++)
		{
			if (sizeof(BackwardType) == 4)
			{ // int32
				reconst_y_float[i] = (static_cast<int32_t>(reconst_y[i])) / (float)(1 << FORWARD_PRECISION);
			}
			else if (sizeof(BackwardType) == 8)
			{ // int64
				reconst_y_float[i] = (static_cast<int64_t>(reconst_y[i])) / (float)(1 << BACKWARD_PRECISION);
			}
			loss += (reconst_y_float[i] - reconst_label_float[i]) * (reconst_y_float[i] - reconst_label_float[i]);
		}
	}
	string loss_func = USE_SOFTMAX_CE ? "Softmax+CE" : "MSE";
	cout << loss_func << " Loss: " << loss / MINI_BATCH_SIZE << endl;
	return loss / MINI_BATCH_SIZE;
}

// original implmentation of NeuralNetwork::getAccuracy(.)
/* void NeuralNetwork::getAccuracy(const RSSVectorMyType &maxIndex, vector<size_t> &counter)
{
	log_print("NN.getAccuracy");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);

	//Needed maxIndex here
	funcMaxpool(outputData, max, maxPrime, rows, columns);

	//Reconstruct things
	RSSVectorMyType temp_max(rows), temp_groundTruth(rows);
	// if (partyNum == PARTY_B)
	// 	sendTwoVectors<RSSMyType>(max, groundTruth, PARTY_A, rows, rows);

	// if (partyNum == PARTY_A)
	// {
	// 	receiveTwoVectors<RSSMyType>(temp_max, temp_groundTruth, PARTY_B, rows, rows);
	// 	addVectors<RSSMyType>(temp_max, max, temp_max, rows);
//		dividePlain(temp_max, (1 << FLOAT_PRECISION));
	// 	addVectors<RSSMyType>(temp_groundTruth, groundTruth, temp_groundTruth, rows);
	// }

	for (size_t i = 0; i < MINI_BATCH_SIZE; ++i)
	{
		counter[1]++;
		if (temp_max[i] == temp_groundTruth[i])
			counter[0]++;
	}

	cout << "Rolling accuracy: " << counter[0] << " out of "
		 << counter[1] << " (" << (counter[0]*100/counter[1]) << " %)" << endl;
} */


void NeuralNetwork::weight_reduction() {
	log_print("NN.weight_reduction");

	// input data reduction
	funcWeightReduction(low_inputData, inputData, inputData.size());

	// each layer weights
	for (size_t i = 0; i < NUM_LAYERS; ++i) {
		// cout << "WR: Layer" << i << endl;
		layers[i]->weight_reduction();
	}
}

void NeuralNetwork::activation_extension() {
	log_print("NN.activation extension");

	for (size_t i = 0; i < NUM_LAYERS; ++i) {
		// cout << "AE: Layer" << i << endl;
		layers[i]->activation_extension();
	}
}

void NeuralNetwork::weight_extension() {
	log_print("NN.weight extension");

	for (size_t i = 0; i < NUM_LAYERS; ++i) {
		// cout << "WE: Layer" << i << endl;
		layers[i]->weight_extension();
	}
}