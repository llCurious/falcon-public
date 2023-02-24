#pragma once
#include "BNLayerOpt.h"
#include "Functionalities.h"
#include <cmath>
using namespace std;

// mixed-precision inverse sqrt. The high-precision inverse sqrt shall be returned. The norm_x used in the backward pass shall be extended.
// low_gamma and low_beta are required in the forward pass.
BNLayerOpt::BNLayerOpt(BNConfig *conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->C, conf->w, conf->h, conf->numBatches),
      gamma(conf->C, make_pair(0, 0)),
      beta(conf->C, make_pair(0, 0)),
    //   inv_sqrt(conf->C),
      high_inv_sqrt(conf->C),
    //   inv_sqrt_rep(conf->numBatches * conf->inputSize),
    //   norm_x(conf->numBatches * conf->C * conf->w * conf->h),
      xmu(conf->numBatches * conf->C * conf->w * conf->h),
      var(conf->C, make_pair(0, 0)),
      high_norm_x(conf->numBatches * conf->C * conf->w * conf->h),
      beta_grad(conf->C, make_pair(0, 0)),
      gamma_grad(conf->C, make_pair(0, 0)),
      activations(conf->numBatches * conf->C * conf->w * conf->h),
      high_activations(conf->numBatches * conf->C * conf->w * conf->h),
    //   low_gamma(conf->C, make_pair(0, 0)),
    //   low_beta(conf->C, make_pair(0, 0)),
      deltas(conf->C * conf->w * conf->h * conf->numBatches),
      beta_velocity(conf->C, make_pair(0, 0)),
	  gamma_velocity(conf->C, make_pair(0, 0))
{
    initialize();
};

void BNLayerOpt::initialize()
{
    if (partyNum == PARTY_A) // gamma = 1
    {
        gamma = BackwardVectorType(conf.C, make_pair(1l << BACKWARD_PRECISION, 0));
    }
    else if (partyNum == PARTY_C)
    {
        gamma = BackwardVectorType(conf.C, make_pair(0, 1l << BACKWARD_PRECISION));
    }
    else
    {
        gamma = BackwardVectorType(conf.C, make_pair(0, 0));
    }
    B = conf.numBatches;
    channel = conf.C;
    width = conf.w;
    height = conf.h;
    size = B * channel * width * height;
};

void BNLayerOpt::printLayer()
{
    cout << "----------------------------------------------" << endl;
    cout << "(" << layerNum + 1 << ") BN Layer\t\t  " << conf.C << " x "
         << conf.numBatches << endl;
}

/**
 * @brief
 *
 * @param inputActivation
 */
void BNLayerOpt::forward(const ForwardVecorType &inputActivation)
{
    // if (MP_TRAINING) {
    //     funcMPBatchNorm(inputActivation, norm_x, inv_sqrt, gamma, beta);
    // } else {

    // }
    funcMPBatchNorm(inputActivation, high_norm_x, xmu, var, high_inv_sqrt, gamma, beta, activations, B, channel, width, height);
    // cout << "forward... " << size << " " << channel << " " << B << " " << endl;
    // ForwardVecorType a = inputActivation;
	// print_vector(a, "FLOAT", "input_BN", a.size());
	// print_vector(weights, "FLOAT", "weights", 100);
	// print_vector(biases, "FLOAT", "biases", biases.size());
	// print_vector(activations, "FLOAT", "out_BN", activations.size());
    /**
    ForwardType eps = (1e-5) * (1l << FORWARD_PRECISION);

    ForwardVecorType var_eps(m, make_pair(0, 0));
    ForwardVecorType mu(m);
    // RSSVectorMyType var_eps(size, make_pair(0, 0)), mu(m, make_pair(0, 0));

    // Compute mean
    for (int i = 0; i < B; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            mu[j] = mu[j] + inputActivation[i * m + j];

        }
    }
    funcProbTruncation<ForwardVecorType, ForwardType>(mu, LOG_MINI_BATCH, m); //  1 truncation by batchSize [1, D]

    // log
    // vector<ForwardType> plainm(m);
    // funcReconstruct(mu, plainm, m, "mean", true);

    // Compute x - mean
    ForwardVecorType x_mean(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            x_mean[i * m + j] = inputActivation[i * m + j] - mu[j];
    // log
    // vector<myType> plainsize(size);
    // funcReconstruct(x_mean, plainsize, size, "x_mean", true);

    // Compute (x-mean)^2
    ForwardVecorType temp2(size);
    funcDotProduct(x_mean, x_mean, temp2, size, true, FORWARD_PRECISION + LOG_MINI_BATCH);

    // mean((x-mean)^2)
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            var_eps[j] = var_eps[j] + temp2[i * m + j];
    // funcReconstruct(var_eps, plainm, m, "var_eps", true);

    // Compute (variance + epsilon)
    funcAddOneConst(var_eps, eps, m);
    // funcReconstruct(var_eps, plainm, m, "var_eps", true);

    // inver Square Root
    // // https://stackoverflow.com/questions/63469333/why-does-the-false-branch-of-if-constexpr-get-compiled
    // if constexpr (MP_FOR_INV_SQRT && std::is_same<decltype(var_eps), RSSVectorLowType>::value) {
    //     cout << "Mixed-Precision Inverse Sqrt" << endl;
	// 	RSSVectorHighType highP_var_eps(m), highP_inv_sqrt(m);
	// 	funcMSExtension(highP_var_eps, var_eps, m);
	// 	funcMulConst(highP_var_eps, highP_var_eps, 1 << (HIGH_PRECISION - LOW_PRECISION), m);	// maintain same precision
    //     funcInverseSqrt<RSSVectorHighType, highBit>(highP_inv_sqrt, highP_var_eps, m); // [1,D]
    //     funcTruncAndReduce(inv_sqrt, highP_inv_sqrt, (HIGH_PRECISION - LOW_PRECISION), m);
    // } else {
    //     funcInverseSqrt<RSSVectorMyType, myType>(inv_sqrt, var_eps, m); // [1,D]
    // }
    mixedPrecisionOp<ForwardVecorType, BackwardVectorType>(inv_sqrt, high_inv_sqrt, var_eps, m);
    // print_vector(inv_sqrt, "FLOAT", "inv_sqrt", inv_sqrt.size());
    // print_vector(var_eps, "FLOAT", "var_eps", 100);
    // print_vector(x_mean, "FLOAT", "x_mean", x_mean.size());

    ForwardVecorType inv_sqrt_rep(size);
    for (int i = 0; i < m; ++i) // compute norm_x
    {
        for (int j = 0; j < B; ++j)
            inv_sqrt_rep[j * m + i] = inv_sqrt[i];
    }
    funcDotProduct<ForwardVecorType, ForwardType>(inv_sqrt_rep, x_mean, norm_x, size, true, FORWARD_PRECISION); // [B, D] * [1, D]
    // print_vector(norm_x, "FLOAT", "norm_x", norm_x.size());

    // self.gamma * self.norm_x + self.beta
    // Scaling
    ForwardVecorType g_repeat(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            g_repeat[i * m + j] = low_gamma[j];
    // print_vector(g_repeat, "FLOAT", "g_repeat", g_repeat.size());
    funcDotProduct(g_repeat, norm_x, activations, size, true, FORWARD_PRECISION);
    // funcReconstruct(activations, plainsize, size, "self.gamma * self.norm_x ", true);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            activations[i * m + j] = activations[i * m + j] + low_beta[j];
    **/
}

// void BNLayerOpt::backward(const RSSVectorMyType &input_grad)
// {
//     cout << "backward... " << size << " " << m << " " << B << " " << endl;
//     //  self.beta_grad = np.sum(grad, axis=0)
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             beta_grad[j] = beta_grad[j] + input_grad[i * m + j];
//     // funcProbTruncation<RSSVectorMyType, myType>(beta_grad, LOG_MINI_BATCH, m);
//     vector<myType> plainm(m);
//     vector<myType> plainsize(size);
//     funcReconstruct(beta_grad, plainm, m, "beta_grad", true);

//     // self.gamma_grad = np.sum(self.norm_x * grad, axis=0)    # 1 multiplication
//     RSSVectorMyType temp(size);
//     funcDotProduct(norm_x, input_grad, temp, size, true, FLOAT_PRECISION);
//     funcReconstruct(temp, plainsize, size, "self.norm_x * grad", true);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             gamma_grad[j] = gamma_grad[j] + temp[i * m + j];
//     funcReconstruct(gamma_grad, plainm, m, "gamma_grad", true);

//     //  dxhat = grad * self.gamma   # 1 multiplication
//     RSSVectorMyType dxhat(size);
//     RSSVectorMyType g_repeat(size);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             g_repeat[i * m + j] = gamma[j];
//     funcDotProduct(input_grad, g_repeat, dxhat, size, true, FLOAT_PRECISION);
//     funcReconstruct(dxhat, plainsize, size, "dxhat", true);

//     // self.act_grad = self.inv_sqrt * (B*dxhat - np.sum(dxhat, axis=0) - self.norm_x * np.sum(dxhat * self.norm_x, axis=0)) / B

//     RSSVectorMyType bdxhat(size);
//     funcMulConst(bdxhat, dxhat, B, size); // B*dxhat
//     funcReconstruct(bdxhat, plainsize, size, "B*dxhat", true);

//     RSSVectorMyType sumdxhat(size, make_pair(0, 0)); // np.sum(dxhat, axis=0)
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             sumdxhat[j] = sumdxhat[j] + dxhat[i * m + j];
//     for (int i = 0; i < m; ++i)
//     {
//         RSSMyType temp = sumdxhat[i];
//         for (int j = 1; j < B; ++j)
//             sumdxhat[j * m + i] = temp;
//     }
//     funcReconstruct(sumdxhat, plainsize, size, "np.sum(dxhat, axis=0)", true);

//     RSSVectorMyType dxx(size); // dxhat * self.norm_x
//     funcDotProduct(dxhat, norm_x, dxx, size, true, FLOAT_PRECISION);
//     funcReconstruct(dxx, plainsize, size, "dxhat * self.norm_x", true);

//     RSSVectorMyType sumdxx(size, make_pair(0, 0)); // np.sum(dxhat * self.norm_x, axis=0)
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             sumdxx[j] = sumdxx[j] + dxx[i * m + j];
//     for (int i = 0; i < m; ++i)
//     {
//         RSSMyType temp = sumdxx[i];
//         for (int j = 1; j < B; ++j)
//             sumdxx[j * m + i] = temp;
//     }
//     funcReconstruct(sumdxx, plainsize, size, "np.sum(dxhat * self.norm_x, axis=0)", true);
//     funcDotProduct(norm_x, sumdxx, sumdxx, size, true, FLOAT_PRECISION);
//     funcReconstruct(sumdxx, plainsize, size, "self.norm_x * np.sum", true);

//     // (bdxhat-sumdxhat-sumdxx)
//     for (size_t i = 0; i < size; i++)
//     {
//         bdxhat[i] = bdxhat[i] - sumdxhat[i] - sumdxx[i];
//     }
//     funcReconstruct(bdxhat, plainsize, size, "(--)", true);
//     // self.inv_sqrt * ()/B
//     // funcReconstruct(inv_sqrt_rep, plainsize, size, "inv_sqrt", true);

//     RSSVectorMyType inv_sqrt_rep(size);
//     for (int i = 0; i < m; ++i) //
//     {
//         for (int j = 0; j < B; ++j)
//             inv_sqrt_rep[j * m + i] = inv_sqrt[i];
//     }
//     funcDotProduct(inv_sqrt_rep, bdxhat, activations, size, true, FLOAT_PRECISION + LOG_MINI_BATCH);
//     // funcReconstruct(activations, plainsize, size, "act_grad", true);
//     print_vector(activations, "FLOAT", "BN Backward- X", size);
// }

// https://kevinzakka.github.io/2016/09/14/batch_normalization/
// void BNLayerOpt::computeDelta(BackwardVectorType &prevDelta)
// {
//     size_t m = channel * width * height;
//     // cout << "BN.computeDelta" << endl;

//     //  dxhat = grad * self.gamma   # 1 multiplication
//     BackwardVectorType dxhat(size);
//     BackwardVectorType g_repeat(size);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < channel; ++j)
// 			for (int k = 0; k < width * height; k++)
//             	g_repeat[i * m + j * width * height + k] = gamma[j];

//     funcDotProduct(g_repeat, deltas, dxhat, size, true, BACKWARD_PRECISION);
//     // funcReconstruct(dxhat, plainsize, size, "dxhat", true);

//     // self.act_grad = self.inv_sqrt * (B*dxhat - np.sum(dxhat, axis=0) - self.norm_x * np.sum(dxhat * self.norm_x, axis=0)) / B
//     BackwardVectorType bdxhat(size);
//     funcMulConst(bdxhat, dxhat, B, size); // B*dxhat
//     // print_vector(dxhat, "FLOAT", "dxhat", dxhat.size());
//     // print_vector(bdxhat, "FLOAT", "bdxhat", bdxhat.size());

//     BackwardVectorType sumdxhat(size, make_pair(0, 0)); // np.sum(dxhat, axis=0)
//     BackwardVectorType sumdxhat_temp(channel, make_pair(0, 0)); // np.sum(dxhat, axis=0)
//     // for (int i = 0; i < B; ++i)
//     //     for (int j = 0; j < m; ++j)
//     //         sumdxhat[j] = sumdxhat[j] + dxhat[i * m + j];
//     // for (int i = 0; i < m; ++i)
//     // {
//     //     RSSBackwardType temp = sumdxhat[i];
//     //     for (int j = 1; j < B; ++j)
//     //         sumdxhat[j * m + i] = temp;
//     // }
//     for (int i = 0; i < channel; ++i)
//     {	
//         for (int j = 0; j < B; ++j)
//         {
// 			for (int k = 0; k < width * height; k++) {
// 				sumdxhat_temp[i] = sumdxhat_temp[i] + dxhat[j * m + i * width * height + k];
// 			}			
//         }
//     }
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < channel; ++j)
// 			for (int k = 0; k < width * height; k++)
//             	sumdxhat[i * m + j * width * height + k] = sumdxhat_temp[j];
//     // print_vector(sumdxhat, "FLOAT", "sumdxhat", sumdxhat.size());


//     BackwardVectorType dxx(size); // dxhat * self.norm_x
//     funcDotProduct(dxhat, high_norm_x, dxx, size, true, BACKWARD_PRECISION);
//     // funcReconstruct(dxx, plainsize, size, "dxhat * self.norm_x", true);

//     BackwardVectorType sumdxx(size, make_pair(0, 0)); // np.sum(dxhat * self.norm_x, axis=0)
//     BackwardVectorType sumdxx_temp(channel, make_pair(0, 0)); // np.sum(dxhat * self.norm_x, axis=0)
//     for (int i = 0; i < channel; ++i)
//     {	
//         for (int j = 0; j < B; ++j)
//         {
// 			for (int k = 0; k < width * height; k++) {
// 				sumdxx_temp[i] = sumdxx_temp[i] + dxx[j * m + i * width * height + k];
// 			}			
//         }
//     }
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < channel; ++j)
// 			for (int k = 0; k < width * height; k++)
//             	sumdxx[i * m + j * width * height + k] = sumdxx_temp[j];
//     // for (int i = 0; i < B; ++i)
//     //     for (int j = 0; j < m; ++j)
//     //         sumdxx[j] = sumdxx[j] + dxx[i * m + j];
//     // for (int i = 0; i < m; ++i)
//     // {
//     //     RSSBackwardType temp = sumdxx[i];
//     //     for (int j = 1; j < B; ++j)
//     //         sumdxx[j * m + i] = temp;
//     // }
//     // print_vector(sumdxx, "FLOAT", "sumdxx", sumdxx.size());

//     funcDotProduct(high_norm_x, sumdxx, sumdxx, size, true, BACKWARD_PRECISION);
//     // print_vector(sumdxx, "FLOAT", "sumdxx", sumdxx.size());


//     // (bdxhat-sumdxhat-sumdxx)
//     for (size_t i = 0; i < size; i++)
//     {
//         bdxhat[i] = bdxhat[i] - sumdxhat[i] - sumdxx[i];
//     }
//     // print_vector(bdxhat, "FLOAT", "bdxhat", bdxhat.size());

//     // self.inv_sqrt * ()/B
//     BackwardVectorType high_inv_sqrt_rep(size);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < channel; ++j)
// 			for (int k = 0; k < width * height; k++)
//             	high_inv_sqrt_rep[i * m + j * width * height + k] = high_inv_sqrt[j];
//     // for (int i = 0; i < m; ++i) //
//     // {
//     //     for (int j = 0; j < B; ++j)
//     //         high_inv_sqrt_rep[j * m + i] = high_inv_sqrt[i];
//     // }
    
//     // print_vector(high_inv_sqrt_rep, "FLOAT", "high_inv_sqrt_rep", high_inv_sqrt_rep.size());
//     // print_vector(high_norm_x, "FLOAT", "high_norm_x", high_norm_x.size());
//     funcDotProduct(high_inv_sqrt_rep, bdxhat, prevDelta, size, true, BACKWARD_PRECISION + LOG_MINI_BATCH);
//     // print_vector(deltas, "FLOAT", "delta", 100);
//     // print_vector(prevDelta, "FLOAT", "x_grad", 100);
//     // prevDelta = deltas; // Test
// }

// updated version: https://stackoverflow.com/questions/67968913/derivative-of-batchnorm2d-in-pytorch
void BNLayerOpt::computeDelta(BackwardVectorType &prevDelta)
{
    size_t m = channel * width * height;

    BackwardVectorType g_repeat(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < channel; ++j)
			for (int k = 0; k < width * height; k++)
            	g_repeat[i * m + j * width * height + k] = gamma[j];

    //  dxhat = grad * self.gamma   # 1 multiplication
    BackwardVectorType dxhat(size);
    funcDotProduct(g_repeat, deltas, dxhat, size, true, BACKWARD_PRECISION);
    // funcReconstruct(dxhat, plainsize, size, "dxhat", true);

    // divar = sum(dxhat*x_mean)
    BackwardVectorType divar_temp(channel);
    BackwardVectorType divar(size);
    funcDotProduct(dxhat, xmu, divar, size, true, BACKWARD_PRECISION);
    for (int i = 0; i < channel; ++i)
    {	
        for (int j = 0; j < B; ++j)
        {
			for (int k = 0; k < width * height; k++) {
				divar_temp[i] = divar_temp[i] + divar[j * m + i * width * height + k];
			}			
        }
    }
    // print_vector(divar, "FLOAT", "dxhat*x_mean", size);
    // print_vector(divar_temp, "FLOAT", "sum(dxhat*x_mean)", channel);

    for (int i = 0; i < B; ++i)
        for (int j = 0; j < channel; ++j)
			for (int k = 0; k < width * height; k++)
            	divar[i * m + j * width * height + k] = divar_temp[j];

    // 1/var
    // BackwardVectorType inv_var(channel);
    // funcReciprocal2(inv_var, var, false, channel);

    BackwardVectorType inv_var_mul_divar(channel);
    BackwardVectorType inv_var_mul_divar_rep(size);
    
    /**
     * Compute 1/var using inv_sqrt * inv_sqrt
     * **/
    BackwardVectorType inv_var(channel);
    funcDotProduct(high_inv_sqrt, high_inv_sqrt, inv_var, channel, true, BACKWARD_PRECISION);  // 1/var
    funcDotProduct(divar_temp, inv_var, inv_var_mul_divar, channel, true, BACKWARD_PRECISION);  // -dsqrtvar

    // funcDivisionByNR(inv_var_mul_divar, divar_temp, var, channel);  // -dsqrtvar
	// print_vector(inv_var, "FLOAT", "reciprocal output", inv_var.size());


    for (int i = 0; i < B; ++i)
        for (int j = 0; j < channel; ++j)
			for (int k = 0; k < width * height; k++)
            	inv_var_mul_divar_rep[i * m + j * width * height + k] = inv_var_mul_divar[j];

    // dvar = -0.5 * divar * 1/var * inv_sqrt
    // dvar*xmu = -0.5 * divar * 1/var * inv_sqrt *xmu = -0.5 * divar * 1/var * norm_x
    // 2dvar*xmu = - divar * 1/var * norm_x
    // BackwardVectorType temp1(size);
    // funcDotProduct(divar, high_norm_x, temp1, size, true, BACKWARD_PRECISION);
    BackwardVectorType dvar_xmu_mul2(size);
    funcDotProduct(inv_var_mul_divar_rep, high_norm_x, dvar_xmu_mul2, size, true, BACKWARD_PRECISION);  // -dxmu2 * B

    BackwardVectorType sum_dvar_xmu_mul2(channel);  // -sum(dxmu2)
    for (int i = 0; i < channel; ++i)
    {	
        for (int j = 0; j < B; ++j)
        {
			for (int k = 0; k < width * height; k++) {
				sum_dvar_xmu_mul2[i] = sum_dvar_xmu_mul2[i] + dvar_xmu_mul2[j * m + i * width * height + k];
			}
        }
    }
    
    // invsqrt_dxhat = invsqrt * dxhat
    BackwardVectorType invsqrt_dxhat(size);
    BackwardVectorType inv_sqrt_rep(size);
	for (int i = 0; i < B; ++i)
        for (int j = 0; j < channel; ++j)
			for (int k = 0; k < width * height; k++)
            	inv_sqrt_rep[i * m + j * width * height + k] = high_inv_sqrt[j];
    funcDotProduct(dxhat, inv_sqrt_rep, invsqrt_dxhat, size, true, BACKWARD_PRECISION); // dxmu1

    BackwardVectorType sum_invsqrt_dxhat(channel);  // sum(dxmu1)
    for (int i = 0; i < channel; ++i)
    {	
        for (int j = 0; j < B; ++j)
        {
			for (int k = 0; k < width * height; k++) {
				sum_invsqrt_dxhat[i] = sum_invsqrt_dxhat[i] + invsqrt_dxhat[j * m + i * width * height + k];
			}
        }
    }


    int dim_bits = int(log2(B * width * height));
    // cout << m << ": " << dim_bits << endl;

    funcProbTruncation<BackwardVectorType, BackwardType>(dvar_xmu_mul2, dim_bits, size);    // -dxmu2

    BackwardVectorType sum_trunc_dvar_xmu(channel, make_pair(0, 0));        // -sum(dxmu2)
    for (int i = 0; i < channel; ++i)
    {	
        for (int j = 0; j < B; ++j)
        {
			for (int k = 0; k < width * height; k++) {
				sum_trunc_dvar_xmu[i] = sum_trunc_dvar_xmu[i] + dvar_xmu_mul2[j * m + i * width * height + k];
			}
        }
    }

    // finalize output
    // davg = davg - sum_invsqrt_dxhat - sum_dvar_xmu_mul2 / m;
    BackwardVectorType davg(channel, make_pair(0, 0));                      // dmu
    for (int i = 0; i < channel; i++) {
        davg[i] = davg[i] - sum_invsqrt_dxhat[i] + sum_trunc_dvar_xmu[i];
    }


    funcProbTruncation<BackwardVectorType, BackwardType>(davg, dim_bits, channel);  // dx2
    BackwardVectorType trunc_davg_rep(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < channel; ++j)
			for (int k = 0; k < width * height; k++)
            	trunc_davg_rep[i * m + j * width * height + k] = davg[j];


    for (int i = 0; i < size; i++) {
        prevDelta[i] = invsqrt_dxhat[i] - dvar_xmu_mul2[i] + trunc_davg_rep[i];
    }
    // print_vector(deltas, "FLOAT", "delta-BN", deltas.size());
    // print_vector(prevDelta, "FLOAT", "BN-prevDelta", 100);
    // print_vector(prevDelta, "FLOAT", "x_grad", 100);
    // prevDelta = deltas; // Test
}

void BNLayerOpt::updateEquations(const BackwardVectorType &prevActivations)
{
    log_print("BN.updateEquations");

    size_t B = conf.numBatches;
    size_t m = channel * width * height;
    // cout << B << " : " << m << endl;

    // print_vector(deltas, "FLOAT", "BN deltas", 100);

    //  Update beta
    //  self.beta_grad = np.sum(grad, axis=0)
    beta_grad = BackwardVectorType(channel, make_pair(0, 0));
    for (int i = 0; i < channel; ++i)
    {	
        for (int j = 0; j < B; ++j)
        {
			for (int k = 0; k < width * height; k++) {
				beta_grad[i] = beta_grad[i] + deltas[j * m + i * width * height + k];
			}			
        }
    }
    // for (int i = 0; i < B; ++i)
    //     for (int j = 0; j < m; ++j)
    //         beta_grad[j] = beta_grad[j] + deltas[i * m + j];
    // funcProbTruncation<RSSVectorMyType, myType>(beta_grad, LOG_MINI_BATCH, m);
    // vector<myType> plainm(m);
    // vector<myType> plainsize(size);
    // funcReconstruct(beta_grad, plainm, m, "beta_grad", true);

    // print_vector(beta_grad_lr, "FLOAT", "beta_grad_lr", 100);

    // BackwardVectorType beta_grad_lr(channel);
    // print_vector(beta_grad, "FLOAT", "BN_bias grad", beta_grad.size());
    // print_vector(beta_grad, "FLOAT", "BN_bias grad", beta_grad.size());

    funcProbTruncation<BackwardVectorType, BackwardType>(beta_grad, LOG_MINI_BATCH + LOG_LEARNING_RATE, channel);
    subtractVectors(beta, beta_grad, beta, channel);
    if (USE_MOMENTUM) {
		// update bias velocity. v' = v * m
		BackwardVectorType diff(beta_velocity.size(), std::make_pair(0, 0));
		funcMulConst(diff, beta_velocity, MOMENTUM, beta_velocity.size());
		funcProbTruncation<BackwardVectorType, BackwardType>(diff, MOMENTUM_BASE, beta_velocity.size());
		// v = v' + g
		addVectors(diff, beta_grad, beta_velocity, beta_velocity.size());
		subtractVectors(beta, diff, beta, beta_velocity.size());
	}

    // funcTruncate(beta_grad, LOG_LEARNING_RATE, m);
    // subtractVectors(beta, beta_grad, beta, m);

    // print_vector(beta_grad, "FLOAT", "BN_bias grad_lr", beta_grad.size());


    // Update gamma
    // self.gamma_grad = np.sum(self.norm_x * grad, axis=0)    # 1 multiplication
    gamma_grad = BackwardVectorType(channel, make_pair(0, 0));
    BackwardVectorType temp(size);
    funcDotProduct(high_norm_x, deltas, temp, size, true, BACKWARD_PRECISION);
    // funcReconstruct(temp, plainsize, size, "self.norm_x * grad", true);

    for (int i = 0; i < channel; ++i)
    {	
        for (int j = 0; j < B; ++j)
        {
			for (int k = 0; k < width * height; k++) {
				gamma_grad[i] = gamma_grad[i] + temp[j * m + i * width * height + k];
			}			
        }
    }
    // print_vector(gamma_grad, "FLOAT", "BN_weight grad", gamma_grad.size());
    // for (int i = 0; i < B; ++i)
    //     for (int j = 0; j < m; ++j)
    //         gamma_grad[j] = gamma_grad[j] + temp[i * m + j];
    // BackwardVectorType gamma_grad_lr(channel);
    funcProbTruncation<BackwardVectorType, BackwardType>(gamma_grad, LOG_MINI_BATCH + LOG_LEARNING_RATE, channel);
    // print_vector(gamma_grad, "FLOAT", "BN_weight grad", gamma_grad.size());
    // funcReconstruct(, plainm, m, "gamma_grad", true);

    subtractVectors(gamma, gamma_grad, gamma, channel);
    if (USE_MOMENTUM) {
		// update bias velocity. v' = v * m
		BackwardVectorType diff(gamma_velocity.size(), std::make_pair(0, 0));
		funcMulConst(diff, gamma_velocity, MOMENTUM, gamma_velocity.size());
		funcProbTruncation<BackwardVectorType, BackwardType>(diff, MOMENTUM_BASE, gamma_velocity.size());
		// v = v' + g
		addVectors(diff, gamma_grad, gamma_velocity, gamma_velocity.size());
		subtractVectors(gamma, diff, gamma, gamma_velocity.size());
	}
}

void BNLayerOpt::weight_reduction() {
	// funcWeightReduction(low_gamma, gamma, gamma.size());
    // funcWeightReduction(low_beta, beta, beta.size());
}

void BNLayerOpt::activation_extension() {
	funcActivationExtension(high_activations, activations, activations.size());
    // funcActivationExtension(high_inv_sqrt, inv_sqrt, inv_sqrt.size());
    // funcActivationExtension(high_norm_x, norm_x, norm_x.size());
}

void BNLayerOpt::weight_extension() {
	// cout << "Not implemented weight extension" << endl;
}