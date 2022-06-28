
#pragma once
#include "NeuralNetConfig.h"
#include "Layer.h"
#include "globals.h"
using namespace std;

class NeuralNetwork
{
public:
	BackwardVectorType inputData;
	ForwardVecorType low_inputData;
	BackwardVectorType outputData;
	BackwardVectorType softmax_output;
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

	// Mixed-precision funcs
	void weight_reduction();
	void activation_extension();
	void weight_extension();
};