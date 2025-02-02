
#pragma once
#include "FCLayer.h"
#include "Functionalities.h"
using namespace std;

FCLayer::FCLayer(FCConfig *conf, int _layerNum)
	: Layer(_layerNum),
	  conf(conf->inputDim, conf->batchSize, conf->outputDim),
	  activations(conf->batchSize * conf->outputDim),
	  high_activations(conf->batchSize * conf->outputDim),
	  deltas(conf->batchSize * conf->outputDim),
	  weights(conf->inputDim * conf->outputDim),
	  low_weights(conf->inputDim * conf->outputDim),
	  biases(conf->outputDim),
	  low_biases(conf->outputDim)
{
	initialize();
}

void FCLayer::initialize()
{
	// Initialize weights and biases here.
	// Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 100;
	size_t size = weights.size();

	// float temp[size];
	// for (size_t i = 0; i < size; ++i){
	// 	temp[i] = (float)(rand() % (higher - lower) + lower)/decimation;

	// 	if (partyNum == PARTY_A){
	// 		weights[i].first = floatToMyType(temp[i]);
	// 		weights[i].second = 0;
	// 	}

	// 	if (partyNum == PARTY_B){
	// 		weights[i].first = 0;
	// 		weights[i].second = 0;
	// 	}

	// 	if (partyNum == PARTY_C){
	// 		weights[i].first = 0;
	// 		weights[i].second = floatToMyType(temp[i]);
	// 	}
	// }

	// fill(biases.begin(), biases.end(), make_pair(0,0));

	/**
	 * Updates from https://github.com/HuangPZ/falcon-public/blob/master/src/FCLayer.cpp
	 * The problem seems to be the initialization to myType has bugs.
	 * TODO: We shall look at this. Since we need both master weights and forward weights.
	 * */
	// RSSVectorMyType temp(size);
	srand(10);
	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; ++i)
		{
			weights[i].first = floatToBackwardType(((float)(rand() % (higher - lower)) - (higher - lower) / 2) / decimation);
			weights[i].second = 0;
		}
		for (size_t i = 0; i < biases.size(); ++i)
		{
			biases[i].first = floatToBackwardType(((float)(rand() % (higher - lower)) - (higher - lower) / 2) / decimation);
			biases[i].second = 0;
		}
	}
	if (partyNum == PARTY_B)
	{
		for (size_t i = 0; i < size; ++i)
		{
			weights[i].first = 0;
			weights[i].second = 0;
		}
		for (size_t i = 0; i < biases.size(); ++i)
		{
			biases[i].first = 0;
			biases[i].second = 0;
		}
	}
	if (partyNum == PARTY_C)
	{
		for (size_t i = 0; i < size; ++i)
		{
			weights[i].second = floatToBackwardType(((float)(rand() % (higher - lower)) - (higher - lower) / 2) / decimation);
			weights[i].first = 0;
		}
		for (size_t i = 0; i < biases.size(); ++i)
		{
			biases[i].second = floatToBackwardType(((float)(rand() % (higher - lower)) - (higher - lower) / 2) / decimation);
			biases[i].first = 0;
		}
	}
}

void FCLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;
	cout << "(" << layerNum + 1 << ") FC Layer\t\t  " << conf.inputDim << " x " << conf.outputDim << endl
		 << "\t\t\t  "
		 << conf.batchSize << "\t\t (Batch Size)" << endl;
}

void FCLayer::forward(const ForwardVecorType &inputActivation)
{
	log_print("FC.forward");

	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows * columns;

	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul<ForwardVecorType>, inputActivation, low_weights, activations, rows, common_dim, columns, 0, 0, FORWARD_PRECISION) << endl;
	else
		funcMatMul(inputActivation, low_weights, activations, rows, common_dim, columns, 0, 0, FORWARD_PRECISION);

	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < columns; ++c)
			activations[r * columns + c] = activations[r * columns + c] + low_biases[c];

	// ForwardVecorType a = inputActivation;
	// print_vector(a, "FLOAT", "input_act", 100);
	// print_vector(weights, "FLOAT", "weights", 100);
	// print_vector(biases, "FLOAT", "biases", biases.size());
	// print_vector(activations, "FLOAT", "out_activations", 100);
}

void FCLayer::computeDelta(BackwardVectorType &prevDelta)
{
	log_print("FC.computeDelta");

	// Back Propagate
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t common_dim = conf.outputDim;

	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul<BackwardVectorType>, deltas, weights, prevDelta, rows, common_dim, columns, 0, 1, BACKWARD_PRECISION) << endl;
	else
		funcMatMul(deltas, weights, prevDelta, rows, common_dim, columns, 0, 1, BACKWARD_PRECISION);
	// print_vector(deltas, "FLOAT", "deltas-" + to_string(deltas.size()), deltas.size());
	// print_vector(prevDelta, "FLOAT", "prevDelta", prevDelta.size());
}

void FCLayer::updateEquations(const BackwardVectorType &prevActivations)
{
	log_print("FC.updateEquations");

	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows * columns;
	BackwardVectorType temp(columns, std::make_pair(0, 0));

	// Update Biases
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			temp[j] = temp[j] + deltas[i * columns + j];
	// TODO-trunc
	if (IS_FALCON)
	{
		funcTruncate(temp, LOG_MINI_BATCH + LOG_LEARNING_RATE, columns);
	}
	else
	{
		funcProbTruncation<BackwardVectorType, BackwardType>(temp, LOG_MINI_BATCH + LOG_LEARNING_RATE, columns);
	}
	subtractVectors(biases, temp, biases, columns);

	// Update Weights
	rows = conf.inputDim;
	columns = conf.outputDim;
	common_dim = conf.batchSize;
	size = rows * columns;
	BackwardVectorType deltaWeight(size);

	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul<BackwardVectorType>, prevActivations, deltas, deltaWeight, rows, common_dim, columns, 1, 0, BACKWARD_PRECISION + LOG_LEARNING_RATE + LOG_MINI_BATCH) << endl;
	else
		funcMatMul(prevActivations, deltas, deltaWeight, rows, common_dim, columns, 1, 0,
				   BACKWARD_PRECISION + LOG_LEARNING_RATE + LOG_MINI_BATCH);

	subtractVectors(weights, deltaWeight, weights, size);
	// cout << "===============================" << endl;
	// RSSVectorMyType xx = prevActivations;
	// print_vector(xx, "FLOAT", "prevActivations", 100);
	// print_vector(deltas, "FLOAT", "deltas-FC", 100);
	// print_vector(deltaWeight, "FLOAT", "deltaWeight-FC", 100);
	// print_vector(temp, "FLOAT", "deltaBias", temp.size());
}


void FCLayer::weight_reduction() {
	funcWeightReduction(low_weights, weights, weights.size());
	funcWeightReduction(low_biases, biases, biases.size());
}

void FCLayer::activation_extension() {
	funcActivationExtension(high_activations, activations, activations.size());
}

void FCLayer::weight_extension() {
	// cout << "Not implemented weight extension" << endl;
}