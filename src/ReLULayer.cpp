
#pragma once
#include "ReLULayer.h"
#include "Functionalities.h"
using namespace std;

ReLULayer::ReLULayer(ReLUConfig* conf, int _layerNum)
:Layer(_layerNum),
 conf(conf->inputDim, conf->batchSize),
 activations(conf->batchSize * conf->inputDim), 
 high_activations(conf->batchSize * conf->inputDim), 
 deltas(conf->batchSize * conf->inputDim),
 reluPrime(conf->batchSize * conf->inputDim)
{}


void ReLULayer::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << layerNum+1 << ") ReLU Layer\t\t  " << conf.batchSize << " x " << conf.inputDim << endl;
}


void ReLULayer::forward(const ForwardVecorType &inputActivation)
{
	log_print("ReLU.forward");

	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

	if (FUNCTION_TIME)
		cout << "funcRELU: " << funcTime(funcRELU<ForwardVecorType>, inputActivation, reluPrime, activations, size) << endl;
	else
		funcRELU(inputActivation, reluPrime, activations, size);
}


void ReLULayer::computeDelta(BackwardVectorType& prevDelta)
{
	log_print("ReLU.computeDelta");

	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

	if (FUNCTION_TIME)
		cout << "funcSelectShares: " << funcTime(static_cast<void(*)(const BackwardVectorType &, const RSSVectorSmallType &,
					  BackwardVectorType &, size_t)>(funcSelectShares), deltas, reluPrime, prevDelta, size) << endl;
	else
		funcSelectShares(deltas, reluPrime, prevDelta, size);
	// print_vector(deltas, "FLOAT", "deltas-ReLU", 100);
}


void ReLULayer::updateEquations(const BackwardVectorType& prevActivations)
{
	log_print("ReLU.updateEquations");
}

void ReLULayer::weight_reduction() {
	// funcWeightReduction(low_weights, weights, weights.size());
	// funcWeightReduction(low_biases, biases, biases.size());
}

void ReLULayer::activation_extension() {
	// cout << "<<< ReLU >>> No need to perform activation extension." << endl;
	// funcActivationExtension(high_activations, activations, activations.size());
}

void ReLULayer::weight_extension() {
	// cout << "Not implemented weight extension" << endl;
}