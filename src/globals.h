
#ifndef GLOBALS_H
#define GLOBALS_H

#pragma once
#include <vector>
#include <string>
#include <assert.h>
#include <limits.h>
#include <array>
#include <bitset>

/********************* Macros *********************/
#define _aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define _aligned_free free
#define getrandom(min, max) ((rand() % (int)(((max) + 1) - (min))) + (min))
#define floatToMyType(a) ((myType)(int)floor(a * (1 << FLOAT_PRECISION)))
#define floatToLowType(a) ((lowBit)(int)floor(a * (1 << FLOAT_PRECISION)))
#define floatToHighType(a) ((highBit)(int)floor(a * (1 << FLOAT_PRECISION)))

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
#define FLOAT_PRECISION 13
#define FLOAT_BIAS (1 << FLOAT_PRECISION)
#define PRECISE_DIVISION false

/********************* Neural Network globals *********************/
// Batch size has to be a power of two
#define LOG_MINI_BATCH 7
#define MINI_BATCH_SIZE (1 << LOG_MINI_BATCH)
#define LOG_LEARNING_RATE 5
#define LEARNING_RATE (1 << (FLOAT_PRECISION - LOG_LEARNING_RATE))
#define NO_OF_EPOCHS 1.5
#define NUM_ITERATIONS 1
// #define NUM_ITERATIONS ((int) (NO_OF_EPOCHS * TRAINING_DATA_SIZE/MINI_BATCH_SIZE))

/********************* Typedefs and others *********************/
typedef uint32_t myType;
typedef uint8_t smallType;
typedef std::pair<myType, myType> RSSMyType;
typedef std::pair<smallType, smallType> RSSSmallType;
typedef std::vector<RSSMyType> RSSVectorMyType;
typedef std::vector<RSSSmallType> RSSVectorSmallType;

// bool
typedef std::pair<bool, bool> RSSBoolType;
typedef std::vector<RSSBoolType> RSSVectorBoolType;

/********************* Quantized Training Types *********************/
typedef uint32_t lowBit;
typedef uint64_t highBit;
typedef std::pair<lowBit, lowBit> RSSLowType;
typedef std::pair<highBit, highBit> RSSHighType;
typedef std::vector<RSSLowType> RSSVectorLowType;
typedef std::vector<RSSHighType> RSSVectorHighType;
const int BIT_SIZE_HIGH = (sizeof(highBit) * CHAR_BIT);
const int BIT_SIZE_LOW = (sizeof(lowBit) * CHAR_BIT);
const lowBit BIT_RANG_LOW = 4294967295;

// Mixed-Precision Setting. Currently, the precision is dependent on the bitwidth.
#define HIGH_PRECISION 20
#define LOW_PRECISION 12

const int BIT_SIZE = (sizeof(myType) * CHAR_BIT);
const myType LARGEST_NEG = ((myType)1 << (BIT_SIZE - 1));       // not used
const myType MINUS_ONE = (myType)-1;                            // wrap computation in tools.h
const smallType BOUNDARY = (256 / PRIME_NUMBER) * PRIME_NUMBER; // AES

/********************* Additional Functions Parameter Setting *********************/
#define EXP_PRECISION 9

/********************* DEBUG AND TEST *********************/
#define DEBUG_ONLY true

#endif
