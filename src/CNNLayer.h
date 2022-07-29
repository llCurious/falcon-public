
#pragma once
#include "CNNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class CNNLayer : public Layer
{
private:
	CNNConfig conf;
	ForwardVecorType activations;
	ForwardVecorType low_weights;
	ForwardVecorType low_biases;
	BackwardVectorType high_activations;
	BackwardVectorType deltas;
	BackwardVectorType weights;
	BackwardVectorType extend_weights;
	BackwardVectorType biases;

public:
	//Constructor and initializer
	CNNLayer(CNNConfig* conf, int _layerNum);
	void initialize();

	//Functions
	void printLayer() override;
	void forward(const ForwardVecorType& inputActivation) override;
	void computeDelta(BackwardVectorType& prevDelta) override;
	void updateEquations(const BackwardVectorType& prevActivations) override;

	// Mixed-precision funcs
	void weight_reduction() override;
	void activation_extension() override;
	void weight_extension() override;

	//Getters
	ForwardVecorType* getActivation() {return &activations;};
	BackwardVectorType* getHighActivation() {return &high_activations;};
	BackwardVectorType* getDelta() {return &deltas;};
	BackwardVectorType* getWeights() {return &weights;};
	BackwardVectorType* getBias() {return &biases;};
};