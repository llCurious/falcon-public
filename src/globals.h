
#ifndef GLOBALS_H
#define GLOBALS_H

#pragma once
#include <vector>
#include <string>
#include <assert.h>
#include <limits.h>
#include <array>
#include <bitset>
#include <iostream>

#define MP_TRAINING true

/********************* Macros *********************/
#define _aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define _aligned_free free
#define getrandom(min, max) ((rand() % (int)(((max) + 1) - (min))) + (min))
#define floatToMyType(a) ((myType)(int)floor(a * (1l << FLOAT_PRECISION)))
#define floatToLowType(a) ((lowBit)(int)floor(a * (1l << LOW_PRECISION)))
#define floatToHighType(a) ((highBit)(long)floor(a * (1l << HIGH_PRECISION)))

/********************* AES and other globals *********************/
#define LOG_DEBUG false
#define LOG_DEBUG_NETWORK false
#define FUNCTION_TIME false
#define RANDOM_COMPUTE 256 // Size of buffer for random elements
#define STRING_BUFFER_SIZE 256
#define PARALLEL true
#define NO_CORES 8

/********************* MPC globals *********************/
#define NUM_OF_PARTIES 3
#define PARTY_A 0
#define PARTY_B 1
#define PARTY_C 2
#define USING_EIGEN false
#define PRIME_NUMBER 67
#define FLOAT_PRECISION 20
#define FLOAT_BIAS (1l << FLOAT_PRECISION)
#define PRECISE_DIVISION false

/********************* Neural Network globals *********************/
// Batch size has to be a power of two
#define REC_ITERS 7 // funcReciprocal use
#define REC_Y 7     // funcReciprocal2 use
#define REC_INIT 4 // This should be 4 for MNIST and CIFAR10, 8 for Tiny ImageNet
#define INVSQRT_ITERS 12
#define LOG_MINI_BATCH 6
#define MINI_BATCH_SIZE (1 << LOG_MINI_BATCH)
// int LOG_LEARNING_RATE = 4;
#define LOG_LEARNING_RATE 4
// #define LEARNING_RATE (1 << (FLOAT_PRECISION - LOG_LEARNING_RATE))
#define NO_OF_EPOCHS 1.5
#define NUM_ITERATIONS 10 // 16000 7820 18750
#define TEST_EVAL_ITERATIONS 1560 // 1500 780 1875
#define TEST_NUM_ITERATIONS 310 // 300 150 310
// #define NUM_ITERATIONS ((int) (NO_OF_EPOCHS * TRAINING_DATA_SIZE/MINI_BATCH_SIZE))

/********************* Typedefs and others *********************/
typedef uint64_t myType;
typedef uint8_t smallType;
typedef std::pair<myType, myType> RSSMyType;
typedef std::pair<smallType, smallType> RSSSmallType;
typedef std::vector<RSSMyType> RSSVectorMyType;
typedef std::vector<RSSSmallType> RSSVectorSmallType;

const int BIT_SIZE = (sizeof(myType) * CHAR_BIT);
const myType LARGEST_NEG = ((myType)1 << (BIT_SIZE - 1));       // not used
const myType MINUS_ONE = (myType)-1;                            // wrap computation in tools.h
const smallType BOUNDARY = (256 / PRIME_NUMBER) * PRIME_NUMBER; // AES

// bool
typedef std::pair<bool, bool> RSSBoolType;
typedef std::vector<RSSBoolType> RSSVectorBoolType;

/********************* Quantized Training Types *********************/
typedef uint32_t lowBit;
typedef uint64_t highBit;
// typedef __int128_t longBit;
typedef unsigned __int128 longBit;
typedef std::pair<lowBit, lowBit> RSSLowType;
typedef std::pair<highBit, highBit> RSSHighType;
typedef std::pair<longBit, longBit> RSSLongType;
typedef std::vector<RSSLowType> RSSVectorLowType;
typedef std::vector<RSSHighType> RSSVectorHighType;
typedef std::vector<RSSLongType> RSSVectorLongType;
const int BIT_SIZE_HIGH = (sizeof(highBit) * CHAR_BIT);
const int BIT_SIZE_LOW = (sizeof(lowBit) * CHAR_BIT);
const highBit BIT_RANG_LOW = 1l << 32;
const longBit longone = 1;

std::ostream &
operator<<(std::ostream &dest, longBit value);

// Mixed-Precision Setting. Currently, the precision is dependent on the bitwidth.
#define HIGH_PRECISION 20
#define LOW_PRECISION 10

/********************* Additional Functions Parameter Setting *********************/
#define EXP_PRECISION 9
#define USE_SOFTMAX_CE true
#define MP_FOR_DIVISION (true && MP_TRAINING)
#define MP_FOR_INV_SQRT (true && MP_TRAINING)
#define MP_FOR_EXP (false && MP_TRAINING)
#define PLAINTEXT_INV_SQRT false
#define PLAINTEXT_RECIPROCAL false
#define PLAINTEXT_EXP false
#define USE_BN true
// denote whether use Mixed-Share Extension or Wrap-Count extension
#define MIXED_SHARE_EXTENSION true

/********************* Mixed-Precision Training Types *********************/
typedef typename std::conditional<MP_TRAINING, RSSVectorHighType, RSSVectorMyType>::type BackwardVectorType;
typedef typename std::conditional<MP_TRAINING, RSSVectorLowType, RSSVectorMyType>::type ForwardVecorType;
typedef typename std::conditional<MP_TRAINING, highBit, myType>::type BackwardType;
typedef typename std::conditional<MP_TRAINING, lowBit, myType>::type ForwardType;
typedef typename std::conditional<MP_TRAINING, RSSHighType, RSSMyType>::type RSSBackwardType;
typedef typename std::conditional<MP_TRAINING, RSSLowType, RSSMyType>::type RSSForwardType;
#if MP_TRAINING
    #define floatToForwardType(a) floatToLowType(a)
    #define floatToBackwardType(a) floatToHighType(a)
    #define FORWARD_PRECISION LOW_PRECISION
    #define BACKWARD_PRECISION HIGH_PRECISION
#else
    #define floatToForwardType(a) floatToMyType(a)
    #define floatToBackwardType(a) floatToMyType(a)
    #define FORWARD_PRECISION FLOAT_PRECISION
    #define BACKWARD_PRECISION FLOAT_PRECISION
#endif

/********************* DEBUG AND TEST *********************/
#define DEBUG_ONLY false
#define OFFLINE_ON false
#define PRE_LOAD true
#define LOAD_TRAINED false   // load from iteration 7812
#define IS_FALCON false
#define USE_GPU true
#define USE_SPDZ_ALEXNET false
#define PORT_START 31000
#define INPUT_TRANSFORM true
#define INPUT_SHUFFLE false
#define USE_MOMENTUM false
#define MOMENTUM_BASE 10
#define MOMENTUM 1 // truncate 4-bit. (12 -> 0.8) (14 -> 0.9)

#endif
