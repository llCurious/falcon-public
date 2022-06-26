
#pragma once
#include "FCConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern int partyNum;


class FCLayer : public Layer
{
private:
	FCConfig conf;
	ForwardVecorType low_weights;
	ForwardVecorType low_biases;
	ForwardVecorType activations;
	BackwardVectorType high_activations;
	BackwardVectorType weights;
	BackwardVectorType biases;
	BackwardVectorType deltas;

public:
	//Constructor and initializer
	FCLayer(FCConfig* conf, int _layerNum);
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