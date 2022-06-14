#pragma once
#include "BNLayerOpt.h"
#include "Functionalities.h"
#include <cmath>
using namespace std;

BNLayerOpt::BNLayerOpt(BNConfig *conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->inputSize, conf->numBatches),
      //   gamma(conf->inputSize, make_pair(0, 0)),
      beta(conf->inputSize, make_pair(0, 0)),
      inv_sqrt(conf->inputSize),
      inv_sqrt_rep(conf->numBatches * conf->inputSize),
      norm_x(conf->numBatches * conf->inputSize),
      beta_grad(conf->inputSize),
      gamma_grad(conf->inputSize, make_pair(0, 0)),
      activations(conf->numBatches * conf->inputSize)
//   xhat(conf->numBatches * conf->inputSize),
//   sigma(conf->numBatches),
//   activations(conf->inputSize * conf->numBatches),
//   deltas(conf->inputSize * conf->numBatches)
{
    initialize();
};

void BNLayerOpt::initialize()
{
    if (partyNum == PARTY_A) // gamma = 1
    {
        gamma = RSSVectorMyType(conf.inputSize, make_pair(1l << FLOAT_PRECISION, 0));
    }
    else if (partyNum == PARTY_C)
    {
        gamma = RSSVectorMyType(conf.inputSize, make_pair(0, 1l << FLOAT_PRECISION));
    }
    else
    {
        gamma = RSSVectorMyType(conf.inputSize, make_pair(0, 0));
    }
    B = conf.numBatches;
    m = conf.inputSize;
    size = B * m;
};

void BNLayerOpt::printLayer()
{
    cout << "----------------------------------------------" << endl;
    cout << "(" << layerNum + 1 << ") BN Layer\t\t  " << conf.inputSize << " x "
         << conf.numBatches << endl;
}

/**
 * @brief
 *
 * @param inputActivation
 */
void BNLayerOpt::forward(const RSSVectorMyType &inputActivation)
{
    cout << "forward... " << size << " " << m << " " << B << " " << endl;
    myType eps = (1e-5) * (1 << FLOAT_PRECISION);

    RSSVectorMyType var_eps(m);
    RSSVectorMyType mu(m);
    // RSSVectorMyType var_eps(size, make_pair(0, 0)), mu(m, make_pair(0, 0));

    // Compute mean
    for (int i = 0; i < B; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            mu[j] = mu[j] + inputActivation[i * m + j];
        }
    }
    funcProbTruncation<RSSVectorMyType, myType>(mu, LOG_MINI_BATCH, m); //  1 truncation by batchSize [1, D]

    // log
    // vector<myType> plainm(m);
    // funcReconstruct(mu, plainm, m, "mean", true);

    // Compute x - mean
    RSSVectorMyType x_mean(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            x_mean[i * m + j] = inputActivation[i * m + j] - mu[j];
    // log
    // vector<myType> plainsize(size);
    // funcReconstruct(x_mean, plainsize, size, "x_mean", true);

    // Compute (x-mean)^2
    RSSVectorMyType temp2(size);
    funcDotProduct(x_mean, x_mean, temp2, size, true, FLOAT_PRECISION + LOG_MINI_BATCH);

    // mean((x-mean)^2)
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            var_eps[j] = var_eps[j] + temp2[i * m + j];
    // funcReconstruct(var_eps, plainm, m, "var_eps", true);

    // Compute (variance + epsilon)
    funcAddOneConst(var_eps, eps, m);
    // funcReconstruct(var_eps, plainm, m, "var_eps", true);

    // inver Square Root
    funcInverseSqrt<RSSVectorMyType, myType>(inv_sqrt, var_eps, m); // [1,D]
    // funcReconstruct(inv_sqrt, plainm, m, "inv_sqrt", true);
    for (int i = 0; i < m; ++i) // scalling invsqrt
    {
        RSSMyType temp = inv_sqrt[i];
        for (int j = 0; j < B; ++j)
            inv_sqrt_rep[j * m + i] = temp;
    }
    // funcReconstruct(inv_sqrt_rep, plainsize, size, "inv_sqrt", true);
    funcDotProduct<RSSVectorMyType, myType>(inv_sqrt_rep, x_mean, norm_x, size, true, FLOAT_PRECISION); // [B, D] * [1, D]
    // funcReconstruct(norm_x, plainsize, size, "norm_x", true);
    // self.gamma * self.norm_x + self.beta
    // Scaling
    RSSVectorMyType g_repeat(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            g_repeat[i * m + j] = gamma[j];
    funcDotProduct(g_repeat, norm_x, activations, size, true, FLOAT_PRECISION);
    // funcReconstruct(activations, plainsize, size, "self.gamma * self.norm_x ", true);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            activations[i * m + j] = activations[i * m + j] + beta[j];
}

void BNLayerOpt::backward(const RSSVectorMyType &input_grad)
{
    cout << "backward... " << size << " " << m << " " << B << " " << endl;
    //  self.beta_grad = np.sum(grad, axis=0)
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            beta_grad[j] = beta_grad[j] + input_grad[i * m + j];
    // funcProbTruncation<RSSVectorMyType, myType>(beta_grad, LOG_MINI_BATCH, m);
    vector<myType> plainm(m);
    vector<myType> plainsize(size);
    funcReconstruct(beta_grad, plainm, m, "beta_grad", true);

    // self.gamma_grad = np.sum(self.norm_x * grad, axis=0)    # 1 multiplication
    RSSVectorMyType temp(size);
    funcDotProduct(norm_x, input_grad, temp, size, true, FLOAT_PRECISION);
    funcReconstruct(temp, plainsize, size, "self.norm_x * grad", true);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            gamma_grad[j] = gamma_grad[j] + temp[i * m + j];
    funcReconstruct(gamma_grad, plainm, m, "gamma_grad", true);

    //  dxhat = grad * self.gamma   # 1 multiplication
    RSSVectorMyType dxhat(size);
    RSSVectorMyType g_repeat(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            g_repeat[i * m + j] = gamma[j];
    funcDotProduct(input_grad, g_repeat, dxhat, size, true, FLOAT_PRECISION);
    funcReconstruct(dxhat, plainsize, size, "dxhat", true);

    // self.act_grad = self.inv_sqrt * (B*dxhat - np.sum(dxhat, axis=0) - self.norm_x * np.sum(dxhat * self.norm_x, axis=0)) / B

    RSSVectorMyType bdxhat(size);
    funcMulConst(bdxhat, dxhat, B, size); // B*dxhat
    funcReconstruct(bdxhat, plainsize, size, "B*dxhat", true);

    RSSVectorMyType sumdxhat(size, make_pair(0, 0)); // np.sum(dxhat, axis=0)
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            sumdxhat[j] = sumdxhat[j] + dxhat[i * m + j];
    for (int i = 0; i < m; ++i)
    {
        RSSMyType temp = sumdxhat[i];
        for (int j = 1; j < B; ++j)
            sumdxhat[j * m + i] = temp;
    }
    funcReconstruct(sumdxhat, plainsize, size, "np.sum(dxhat, axis=0)", true);

    RSSVectorMyType dxx(size); // dxhat * self.norm_x
    funcDotProduct(dxhat, norm_x, dxx, size, true, FLOAT_PRECISION);
    funcReconstruct(dxx, plainsize, size, "dxhat * self.norm_x", true);

    RSSVectorMyType sumdxx(size, make_pair(0, 0)); // np.sum(dxhat * self.norm_x, axis=0)
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            sumdxx[j] = sumdxx[j] + dxx[i * m + j];
    for (int i = 0; i < m; ++i)
    {
        RSSMyType temp = sumdxx[i];
        for (int j = 1; j < B; ++j)
            sumdxx[j * m + i] = temp;
    }
    funcReconstruct(sumdxx, plainsize, size, "np.sum(dxhat * self.norm_x, axis=0)", true);
    funcDotProduct(norm_x, sumdxx, sumdxx, size, true, FLOAT_PRECISION);
    funcReconstruct(sumdxx, plainsize, size, "self.norm_x * np.sum", true);

    // (bdxhat-sumdxhat-sumdxx)
    for (size_t i = 0; i < size; i++)
    {
        bdxhat[i] = bdxhat[i] - sumdxhat[i] - sumdxx[i];
    }
    funcReconstruct(bdxhat, plainsize, size, "(--)", true);
    // self.inv_sqrt * ()/B
    funcDotProduct(inv_sqrt_rep, bdxhat, activations, size, true, FLOAT_PRECISION + LOG_MINI_BATCH);
}

// https://kevinzakka.github.io/2016/09/14/batch_normalization/
// void BNLayerObj::computeDelta(RSSVectorMyType &prevDelta)
// {
//     log_print("BN.computeDelta");

//     size_t B = conf.numBatches;
//     size_t m = conf.inputSize;

//     // Derivative with xhat
//     RSSVectorMyType g_repeat(B * m), dxhat(B * m);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             g_repeat[i * m + j] = gamma[i];

//     funcDotProduct(g_repeat, deltas, dxhat, B * m, true, FLOAT_PRECISION);

//     // First term
//     RSSVectorMyType temp1(B * m);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp1[i * m + j] = ((myType)m) * dxhat[i * m + j];

//     // Second term
//     RSSVectorMyType temp2(B * m, make_pair(0, 0));
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp2[i * m] = temp2[i * m] + dxhat[i * m + j];

//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp2[i * m + j] = temp2[i * m];

//     // Third term
//     RSSVectorMyType temp3(B * m, make_pair(0, 0));
//     funcDotProduct(dxhat, xhat, temp3, B * m, true, FLOAT_PRECISION);
//     for (int i = 0; i < B; ++i)
//         for (int j = 1; j < m; ++j)
//             temp3[i * m] = temp3[i * m] + temp3[i * m + j];

//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp3[i * m + j] = temp3[i * m];

//     funcDotProduct(temp3, xhat, temp3, B * m, true, FLOAT_PRECISION);

//     // Numerator
//     subtractVectors<RSSMyType>(temp1, temp2, temp1, B * m);
//     subtractVectors<RSSMyType>(temp1, temp3, temp1, B * m);

//     RSSVectorMyType temp4(B);
//     for (int i = 0; i < B; ++i)
//         temp4[i] = ((myType)m) * sigma[i];

//     funcBatchNorm(temp1, temp4, prevDelta, m, B);
// }

// void BNLayerObj::updateEquations(const RSSVectorMyType &prevActivations)
// {
//     log_print("BN.updateEquations");

//     size_t B = conf.numBatches;
//     size_t m = conf.inputSize;

//     // Update beta
//     RSSVectorMyType temp1(B, make_pair(0, 0));
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp1[i] = temp1[i] + deltas[i * m + j];

//     subtractVectors<RSSMyType>(beta, temp1, beta, B);

//     // Update gamma
//     RSSVectorMyType temp2(B * m, make_pair(0, 0)), temp3(B, make_pair(0, 0));
//     funcDotProduct(xhat, deltas, temp2, B * m, true, FLOAT_PRECISION);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp3[i] = temp3[i] + temp2[i * m + j];

//     subtractVectors<RSSMyType>(gamma, temp3, gamma, B);
// }
