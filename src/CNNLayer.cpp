#pragma once
#include "CNNLayer.h"
#include "Functionalities.h"
using namespace std;

extern bool LARGE_NETWORK;

CNNLayer::CNNLayer(CNNConfig *conf, int _layerNum)
	: Layer(_layerNum),
	  conf(conf->imageHeight, conf->imageWidth, conf->inputFeatures,
		   conf->filters, conf->filterSize, conf->stride,
		   conf->padding, conf->batchSize),
	  weights(conf->filterSize * conf->filterSize * conf->inputFeatures * conf->filters),
	  extend_weights(conf->filterSize * conf->filterSize * conf->inputFeatures * conf->filters),
	  biases(conf->filters),
	  low_weights(conf->filterSize * conf->filterSize * conf->inputFeatures * conf->filters),
	  low_biases(conf->filters),
	  activations(conf->batchSize * conf->filters *
				  (((conf->imageWidth - conf->filterSize + 2 * conf->padding) / conf->stride) + 1) *
				  (((conf->imageHeight - conf->filterSize + 2 * conf->padding) / conf->stride) + 1)),
	  high_activations(conf->batchSize * conf->filters *
				  (((conf->imageWidth - conf->filterSize + 2 * conf->padding) / conf->stride) + 1) *
				  (((conf->imageHeight - conf->filterSize + 2 * conf->padding) / conf->stride) + 1)),
	  deltas(conf->batchSize * conf->filters *
			 (((conf->imageWidth - conf->filterSize + 2 * conf->padding) / conf->stride) + 1) *
			 (((conf->imageHeight - conf->filterSize + 2 * conf->padding) / conf->stride) + 1))
{
	initialize();
};

void CNNLayer::initialize()
{
	// Initialize weights and biases here.
	// Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 100;
	size_t size = weights.size();

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

void CNNLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;
	cout << "(" << layerNum + 1 << ") CNN Layer\t\t  " << conf.imageHeight << " x " << conf.imageWidth
		 << " x " << conf.inputFeatures << endl
		 << "\t\t\t  "
		 << conf.filterSize << " x " << conf.filterSize << "  \t(Filter Size)" << endl
		 << "\t\t\t  "
		 << conf.stride << " , " << conf.padding << " \t(Stride, padding)" << endl
		 << "\t\t\t  "
		 << conf.batchSize << "\t\t(Batch Size)" << endl
		 << "\t\t\t  "
		 << (((conf.imageWidth - conf.filterSize + 2 * conf.padding) / conf.stride) + 1) << " x "
		 << (((conf.imageHeight - conf.filterSize + 2 * conf.padding) / conf.stride) + 1) << " x "
		 << conf.filters << " \t(Output)" << endl;
}

void CNNLayer::forward(const ForwardVecorType &inputActivation)
{
	log_print("CNN.forward");

	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t f = conf.filterSize;
	size_t Din = conf.inputFeatures;
	size_t Dout = conf.filters;
	size_t P = conf.padding;
	size_t S = conf.stride;
	size_t ow = (((iw - f + 2 * P) / S) + 1);
	size_t oh = (((ih - f + 2 * P) / S) + 1);

	// Reshape activations
	ForwardVecorType temp1((iw + 2 * P) * (ih + 2 * P) * Din * B, make_pair(0, 0));
	// if (FUNCTION_TIME)
	// 	cout << "ZP: \t" << funcTime(zeroPad, inputActivation, temp1, iw, ih, P, Din, B) << endl;
	// else
	zeroPad(inputActivation, temp1, iw, ih, P, Din, B);

	// Reshape for convolution
	ForwardVecorType temp2((f * f * Din) * (ow * oh * B));
	// if (FUNCTION_TIME)
	// 	cout << "convToMult: " << funcTime(convToMult, temp1, temp2, (iw+2*P), (ih+2*P), f, Din, S, B) << endl;
	// else
	// 	convToMult(temp1, temp2, (iw+2*P), (ih+2*P), f, Din, S, B);

	{
		size_t loc_input, loc_output;
		for (size_t i = 0; i < B; ++i)
			for (size_t j = 0; j < oh; j++)
				for (size_t k = 0; k < ow; k++)
				{
					loc_output = (i * ow * oh + j * ow + k);
					for (size_t l = 0; l < Din; ++l)
					{
						loc_input = i * (iw + 2 * P) * (ih + 2 * P) * Din + l * (iw + 2 * P) * (ih + 2 * P) + j * S * (iw + 2 * P) + k * S;
						for (size_t a = 0; a < f; ++a)	   // filter height
							for (size_t b = 0; b < f; ++b) // filter width
								temp2[(l * f * f + a * f + b) * ow * oh * B + loc_output] = temp1[loc_input + a * (iw + 2 * P) + b];
					}
				}
	}

	// Perform the multiplication.
	ForwardVecorType temp3(Dout * (ow * oh * B));
	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul<ForwardVecorType>, low_weights, temp2, temp3, Dout, (f * f * Din), (ow * oh * B), 0, 0, FORWARD_PRECISION) << endl;
	else
		funcMatMul(low_weights, temp2, temp3, Dout, (f * f * Din), (ow * oh * B), 0, 0, FORWARD_PRECISION);

	// Add biases and meta-transpose
	size_t tempSize = ow * oh;
	for (size_t i = 0; i < B; ++i)
		for (size_t j = 0; j < Dout; ++j)
			for (size_t k = 0; k < tempSize; ++k)
				activations[i * Dout * tempSize + j * tempSize + k] = temp3[j * B * tempSize + i * tempSize + k] + low_biases[j];
	
	// ForwardVecorType a = inputActivation;
	// print_vector(a, "FLOAT", "input_cnn", 100);
	// // print_vector(weights, "FLOAT", "weights", 10);
	// // print_vector(biases, "FLOAT", "biases", biases.size());
	// print_vector(activations, "FLOAT", "out_cnn", 100);
}

// TODO: Recheck backprop after forward bug fixed.
void CNNLayer::computeDelta(BackwardVectorType &prevDelta)
{
	log_print("CNN.computeDelta");

	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t f = conf.filterSize;
	size_t Din = conf.inputFeatures;
	size_t Dout = conf.filters;
	size_t P = conf.padding;
	size_t S = conf.stride;
	size_t ow = (((iw - f + 2 * P) / S) + 1);
	size_t oh = (((ih - f + 2 * P) / S) + 1);

	BackwardVectorType temp1((f * f * Dout) * (iw * ih * B), make_pair(0, 0));
	{
		size_t x, y;
		size_t sizeDeltaBeta = iw;
		size_t sizeDeltaB = sizeDeltaBeta * ih;
		size_t sizeDeltaP = sizeDeltaB * B;
		size_t sizeDeltaQ = sizeDeltaP * f;
		size_t sizeDeltaD = sizeDeltaQ * f;

		size_t sizeY = ow;
		size_t sizeD = sizeY * oh;
		size_t sizeB = sizeD * Dout;

		for (int d = 0; d < Dout; ++d)
			for (size_t q = 0; q < f; ++q)
				for (size_t p = 0; p < f; ++p)
					for (int b = 0; b < B; ++b)
						for (size_t beta = 0; beta < ih; ++beta)
							for (size_t alpha = 0; alpha < iw; ++alpha)
								if ((alpha + P - p) % S == 0 and (beta + P - q) % S == 0)
								{
									x = (alpha + P - p) / S;
									y = (beta + P - q) / S;
									if (x >= 0 and x < ow and y >= 0 and y < oh)
									{
										temp1[d * sizeDeltaD + q * sizeDeltaQ + p * sizeDeltaP +
											  b * sizeDeltaB + beta * sizeDeltaBeta + alpha] =
											deltas[b * sizeB + d * sizeD + y * sizeY + x];
									}
								}
	}

	BackwardVectorType temp2((Din) * (f * f * Dout), make_pair(0, 0));
	{
		size_t sizeQ = f;
		size_t sizeR = sizeQ * f;
		size_t sizeD = sizeR * Din;

		size_t sizeWeightsQ = f;
		size_t sizeWeightsD = sizeWeightsQ * f;
		size_t sizeWeightsR = sizeWeightsD * Dout;

		for (int d = 0; d < Dout; ++d)
			for (size_t r = 0; r < Din; ++r)
				for (size_t q = 0; q < f; ++q)
					for (size_t p = 0; p < f; ++p)
					{
						temp2[r * sizeWeightsR + d * sizeWeightsD + q * sizeWeightsQ + p] =
							extend_weights[d * sizeD + r * sizeR + q * sizeQ + p];
					}
	}

	BackwardVectorType temp3((Din) * (iw * ih * B), make_pair(0, 0));

	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul<BackwardVectorType>, temp2, temp1, temp3, Din, (f * f * Dout), (iw * ih * B), 0, 0, BACKWARD_PRECISION) << endl;
	else
		funcMatMul(temp2, temp1, temp3, Din, (f * f * Dout), (iw * ih * B), 0, 0, BACKWARD_PRECISION);

	{
		size_t sizeDeltaBeta = iw;
		size_t sizeDeltaB = sizeDeltaBeta * ih;
		size_t sizeDeltaR = sizeDeltaB * B;

		size_t sizeBeta = iw;
		size_t sizeR = sizeBeta * ih;
		size_t sizeB = sizeR * Din;

		for (int r = 0; r < Din; ++r)
			for (int b = 0; b < B; ++b)
				for (size_t beta = 0; beta < ih; ++beta)
					for (size_t alpha = 0; alpha < iw; ++alpha)
					{
						prevDelta[b * sizeB + r * sizeR + beta * sizeBeta + alpha] =
							temp3[r * sizeDeltaR + b * sizeDeltaB + beta * sizeDeltaBeta + alpha];
					}
	}

	// cout << "CNN delta shape " << deltas.size() << ", CNN prevDelta shape: " << prevDelta.size() << endl;  
	// print_vector(deltas, "FLOAT", "CNN-delta", 100);
	// print_vector(prevDelta, "FLOAT", "CNN-prevDelta", 100);
}

void CNNLayer::updateEquations(const BackwardVectorType &prevActivations)
{
	log_print("CNN.updateEquations");

	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t f = conf.filterSize;
	size_t Din = conf.inputFeatures;
	size_t Dout = conf.filters;
	size_t P = conf.padding;
	size_t S = conf.stride;
	size_t ow = (((iw - f + 2 * P) / S) + 1);
	size_t oh = (((ih - f + 2 * P) / S) + 1);

	/********************** Bias update **********************/
	// Bias update
	BackwardVectorType temp1(Dout, make_pair(0, 0));
	{
		size_t sizeY = ow;
		size_t sizeD = sizeY * oh;
		size_t sizeB = sizeD * Dout;
		for (int d = 0; d < Dout; ++d)
			for (size_t b = 0; b < B; ++b)
				for (size_t y = 0; y < oh; ++y)
					for (size_t x = 0; x < ow; ++x)
						temp1[d] = temp1[d] + deltas[b * sizeB + d * sizeD + y * sizeY + x];
	}
	// cout << "bias shape: " << biases.size() << endl;
	// print_vector(temp1, "FLOAT", "deltaBias-CNN", 100);
	if (IS_FALCON)
	{
		funcTruncate(temp1, LOG_MINI_BATCH + LOG_LEARNING_RATE, Dout);
	}
	else
	{
		funcProbTruncation<BackwardVectorType, BackwardType>(temp1, LOG_MINI_BATCH + LOG_LEARNING_RATE, Dout);
	}
	subtractVectors(biases, temp1, biases, Dout);

	/********************** Weights update **********************/
	// Reshape activations
	BackwardVectorType temp3((f * f * Din) * (ow * oh * B));
	{
		size_t sizeY = ow;
		size_t sizeB = sizeY * oh;
		size_t sizeP = sizeB * B;
		size_t sizeQ = sizeP * f;
		size_t sizeR = sizeQ * f;

		size_t actSizeBeta = iw;
		size_t actSizeR = actSizeBeta * ih;
		size_t actSizeB = actSizeR * Din;

		for (size_t r = 0; r < Din; ++r)
			for (size_t p = 0; p < f; ++p)
				for (size_t q = 0; q < f; ++q)
					for (int b = 0; b < B; ++b)
						for (int y = 0; y < oh; ++y)
							for (int x = 0; x < ow; ++x)
								if ((x * S - P + p) >= 0 and (x * S - P + p) < iw and
									(y * S - P + q) >= 0 and (y * S - P + q) < ih)
								{
									temp3[r * sizeR + q * sizeQ + p * sizeP +
										  b * sizeB + y * sizeY + x] =
										prevActivations[b * actSizeB + r * actSizeR +
														(y * S - P + q) * actSizeBeta + (x * S - P + p)];
								}
	}

	// Reshape delta
	BackwardVectorType temp2((Dout) * (ow * oh * B));
	{
		size_t sizeY = ow;
		size_t sizeD = sizeY * oh;
		size_t sizeB = sizeD * Dout;
		size_t counter = 0;

		for (size_t d = 0; d < Dout; ++d)
			for (int b = 0; b < B; ++b)
				for (int y = 0; y < oh; ++y)
					for (int x = 0; x < ow; ++x)
						temp2[counter++] = deltas[b * sizeB + d * sizeD + y * sizeY + x];
	}

	// Compute product, truncate and subtract
	BackwardVectorType temp4((Dout) * (f * f * Din));
	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul<BackwardVectorType>, temp2, temp3, temp4, (Dout), (ow * oh * B), (f * f * Din), 0, 1, BACKWARD_PRECISION + LOG_MINI_BATCH + LOG_LEARNING_RATE) << endl;
	else
		funcMatMul(temp2, temp3, temp4, (Dout), (ow * oh * B), (f * f * Din), 0, 1,
				   BACKWARD_PRECISION + LOG_MINI_BATCH + LOG_LEARNING_RATE);

	// BackwardVectorType ttt = prevActivations;
	// print_vector(ttt, "FLOAT", "CNN prev act", 100);
	// print_vector(deltas, "FLOAT", "CNN deltas", 100);

	// cout << "CNN weight_grad size: " << temp4.size() << endl;
	// print_vector(temp4, "FLOAT", "CNN weight_grad", 100);
	subtractVectors(weights, temp4, weights, f * f * Din * Dout);
	// print_vector(temp4, "FLOAT", "deltaWeight", 100);
}

void CNNLayer::weight_reduction() {
	funcWeightReduction(low_weights, weights, weights.size());
	funcWeightReduction(low_biases, biases, biases.size());
}

void CNNLayer::activation_extension() {
	funcActivationExtension(high_activations, activations, activations.size());
}

void CNNLayer::weight_extension() {
	// cout << "Not implemented weight extension" << endl;
	funcWeightExtension(extend_weights, weights, weights.size());
}