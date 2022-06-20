
#pragma once
#include "NeuralNetConfig.h"
#include "Layer.h"
#include "globals.h"
using namespace std;

class NeuralNetwork
{
public:
	RSSVectorMyType inputData;
	RSSVectorMyType outputData;
	RSSVectorMyType softmax_output;
	vector<Layer*> layers;

	NeuralNetwork(NeuralNetConfig* config);
	~NeuralNetwork();
	void forward();
	void backward();
	void computeDelta();
	void updateEquations();
	void predict(RSSVectorMyType &maxIndex);
	float getAccuracy();
	float getLoss();
};