
#ifndef TOOLS_H
#define TOOLS_H
#pragma once

#include <stdio.h>
#include <iostream>
#include "Config.h"
#include "../util/TedKrovetzAesNiWrapperC.h"
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <vector>
#include <time.h>
#include "secCompMultiParty.h"
#include "main_gf_funcs.h"
#include <string>
#include <openssl/sha.h>
#include <math.h>
#include <sstream>
#include "AESObject.h"
#include "connect.h"
#include "globals.h"
#include <bitset>

extern int partyNum;

extern AESObject *aes_next;
extern AESObject *aes_indep;

extern smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType subtractModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];

#if MULTIPLICATION_TYPE == 0
#define MUL(x, y) gfmul(x, y)
#define MULT(x, y, ans) gfmul(x, y, &ans)

#ifdef OPTIMIZED_MULTIPLICATION
#define MULTHZ(x, y, ans) gfmulHalfZeros(x, y, &ans) // optimized multiplication when half of y is zeros
#define MULHZ(x, y) gfmulHalfZeros(x, y)			 // optimized multiplication when half of y is zeros
#else
#define MULTHZ(x, y, ans) gfmul(x, y, &ans)
#define MULHZ(x, y) gfmul(x, y)
#endif
#define SET_ONE _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

#else
#define MUL(x, y) gfmul3(x, y)
#define MULT(x, y, ans) gfmul3(x, y, &ans)
#ifdef OPTIMIZED_MULTIPLICATION
#define MULTHZ(x, y, ans) gfmul3HalfZeros(x, y, &ans) // optimized multiplication when half of y is zeros
#define MULHZ(x, y) gfmul3HalfZeros(x, y)			  // optimized multiplication when half of y is zeros
#else
#define MULTHZ(x, y, ans) gfmul3(x, y, &ans)
#define MULHZ(x, y) gfmul3(x, y)
#endif
#define SET_ONE _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
#endif

//
// field zero
#define SET_ZERO _mm_setzero_si128()
// the partynumber(+1) embedded in the field
#define SETX(j) _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, j + 1) // j+1
// Test if 2 __m128i variables are equal
#define EQ(x, y) _mm_test_all_ones(_mm_cmpeq_epi8(x, y))
// Add 2 field elements in GF(2^128)
#define ADD(x, y) _mm_xor_si128(x, y)
// Subtraction and addition are equivalent in characteristic 2 fields
#define SUB ADD
// Evaluate x^n in GF(2^128)
#define POW(x, n) fastgfpow(x, n)
// Evaluate x^2 in GF(2^128)
#define SQ(x) square(x)
// Evaluate x^(-1) in GF(2^128)
#define INV(x) inverse(x)
// Evaluate P(x), where p is a given polynomial, in GF(2^128)
#define EVAL(x, poly, ans) fastPolynomEval(x, poly, &ans) // polynomEval(SETX(x),y,z)
// Reconstruct the secret from the shares
#define RECON(shares, deg, secret) reconstruct(shares, deg, &secret)
// returns a (pseudo)random __m128i number using AES-NI
#define RAND LoadSeedNew
// returns a (pseudo)random bit using AES-NI
#define RAND_BIT LoadBool

// the encryption scheme
#define PSEUDO_RANDOM_FUNCTION(seed1, seed2, index, numberOfBlocks, result) pseudoRandomFunctionwPipelining(seed1, seed2, index, numberOfBlocks, result);

// The degree of the secret-sharings before multiplications
extern int degSmall;
// The degree of the secret-sharing after multiplications (i.e., the degree of the secret-sharings of the PRFs)
extern int degBig;
// The type of honest majority we assume
extern int majType;

// bases for interpolation
extern __m128i *baseReduc;
extern __m128i *baseRecon;
// saved powers for evaluating polynomials
extern __m128i *powers;

// one in the field
extern const __m128i ONE;
// zero in the field
extern const __m128i ZERO;

extern int testCounter;

typedef struct polynomial
{
	int deg;
	__m128i *coefficients;
} Polynomial;

void gfmul(__m128i a, __m128i b, __m128i *res);

// This function works correctly only if all the upper half of b is zeros
void gfmulHalfZeros(__m128i a, __m128i b, __m128i *res);

// multiplies a and b
__m128i gfmul(__m128i a, __m128i b);

// This function works correctly only if all the upper half of b is zeros
__m128i gfmulHalfZeros(__m128i a, __m128i b);

__m128i gfpow(__m128i x, int deg);

__m128i fastgfpow(__m128i x, int deg);

__m128i square(__m128i x);

__m128i inverse(__m128i x);

string _sha256hash_(char *input, int length);

string sha256hash(char *input, int length);

void printError(string error);

string __m128i_toHex(__m128i var);

string __m128i_toString(__m128i var);

__m128i stringTo__m128i(string str);

unsigned int charValue(char c);

string convertBooltoChars(bool *input, int length);

string toHex(string s);

string convertCharsToString(char *input, int size);

void print(__m128i *arr, int size);

void print128_num(__m128i var);

void log_print(string str);
void error(string str);
string which_network(string network);

/************Debug Tools****************/
template <typename V>
bool equalRssVector(const V &v1, const V &v2, size_t size)
{
	assert(v1.size() == v2.size() && v2.size() == size && "size is not consistent");

	for (size_t i = 0; i < size; i++)
	{
		if (v1[i] != v2[i])
			return false;
	}
	return true;
}
// template <typename T, typename B>
// void printOneRss(T var, string type)
// {
// 	if (type == "BITS")
// 		cout << "( " << bitset<BIT_SIZE>(var.fisrt) << ", " << bitset<BIT_SIZE>(var.second) << ") ";
// 	else if (type == "FLOAT")
// 		cout << "(" << (static_cast<B>(var.first)) / (float)(1 << FLOAT_PRECISION) << ", " << (static_cast<B>(var.second)) / (float)(1 << FLOAT_PRECISION) << ") ";
// 	else if (type == "SIGNED")
// 		cout << "(" << (static_cast<B>(var.first)) << ", " << (static_cast<B>(var.second)) << ") ";
// 	else if (type == "UNSIGNED")
// 		cout << "(" << var.first << ", " << var.second << ") ";
// }

void printBoolVec(vector<bool> &data, string str, size_t size);
void printBoolRssVec(const RSSVectorBoolType &data, string str, size_t size);

template <typename Vec>
void printRssVector(Vec &var, string pre_text, int print_nos)
{
	cout << pre_text << " " << print_nos << endl;
	for (size_t i = 0; i < print_nos; i++)
	{
		// cout << "(" << (static_cast<int64_t>(var[i].first)) / (float)(1 << FLOAT_PRECISION) << ", " << (static_cast<int64_t>(var[i].second)) / (float)(1 << FLOAT_PRECISION) << ")" << endl;
		cout << var[i].first << " " << var[i].second << endl;
		// printOneRss<T, B>(var[i], type);
	}
}

void printHighBitVec(vector<highBit> &var, string pre_text, int print_nos);
void printLowBitVec(vector<lowBit> &var, string pre_text, int print_nos);

template <typename T>
void printVector(const vector<T> &var, string pre_text, int print_nos)
{
	cout << pre_text << " " << print_nos << endl;
	for (size_t i = 0; i < print_nos; i++)
	{
		// cout << (static_cast<int64_t>(var[i])) / (float)(1 << FLOAT_PRECISION) << endl;
		cout << var[i] << " ";
	}
	cout << endl;
}

template <typename T>
void print_myType(T var, string message, string type)
{
	if (std::is_same<T, RSSVectorHighType>::value)
	{
		if (type == "BITS")
			cout << message << ": " << bitset<BIT_SIZE>(var) << endl;
		else if (type == "FLOAT")
			cout << message << ": " << (static_cast<int64_t>(var)) / (float)(1 << FLOAT_PRECISION) << endl;
		else if (type == "SIGNED")
			cout << message << ": " << static_cast<int64_t>(var) << endl;
		else if (type == "UNSIGNED")
			cout << message << ": " << var << endl;
	}
	if (std::is_same<T, RSSVectorLowType>::value)
	{
		if (type == "BITS")
			cout << message << ": " << bitset<BIT_SIZE>(var) << endl;
		else if (type == "FLOAT")
			cout << message << ": " << (static_cast<int32_t>(var)) / (float)(1 << FLOAT_PRECISION) << endl;
		else if (type == "SIGNED")
			cout << message << ": " << static_cast<int32_t>(var) << endl;
		else if (type == "UNSIGNED")
			cout << message << ": " << var << endl;
	}
}

template <typename T>
void print_linear(T var, string type);
template <typename T>
void print_linear(T var, string type)
{
	if (std::is_same<T, highBit>::value)
	{
		if (type == "BITS")
			cout << bitset<BIT_SIZE>(var) << " ";
		else if (type == "FLOAT")
			cout << (static_cast<int64_t>(var)) / (float)(1 << HIGH_PRECISION) << " ";
		else if (type == "SIGNED")
			cout << static_cast<int64_t>(var) << " ";
		else if (type == "UNSIGNED")
			cout << var << " ";
	}
	else if (std::is_same<T, lowBit>::value)
	{
		if (type == "BITS")
			cout << bitset<BIT_SIZE>(var) << " ";
		else if (type == "FLOAT")
			cout << (static_cast<int32_t>(var)) / (float)(1 << LOW_PRECISION) << " ";
		else if (type == "SIGNED")
			cout << static_cast<int32_t>(var) << " ";
		else if (type == "UNSIGNED")
			cout << var << " ";
	}
}
// move to Functionalities.h
// extern template void funcReconstruct<RSSVectorHighType, highBit>(const RSSVectorHighType &a, vector<highBit> &b, size_t size, string str, bool print);
// extern template void funcReconstruct<RSSVectorLowType, lowBit>(const RSSVectorLowType &a, vector<lowBit> &b, size_t size, string str, bool print);

// template<typename T>
// void print_vector(T &var, string type, string pre_text, int print_nos) {
// 	size_t temp_size = var.size();
// 	// FIXME
// 	typedef typename std::conditional<std::is_same<T, RSSVectorHighType>::value, highBit, lowBit>::type computeType;
// 	vector<computeType> b(temp_size);
// 	funcReconstruct(var, b, temp_size, "anything", false);
// 	cout << pre_text << endl;
// 	for (int i = 0; i < print_nos; ++i){
// 		print_linear(b[i], type);
// 		// if (i % 10 == 9)
// 			// std::cout << std::endl;
// 	}
// 	cout << endl;
// }

void print_vector(RSSVectorSmallType &var, string type, string pre_text, int print_nos);
template <typename Vec, typename T>
void matrixMultRSS(const Vec &a, const Vec &b, vector<T> &temp3,
				   size_t rows, size_t common_dim, size_t columns,
				   size_t transpose_a, size_t transpose_b);
template <typename Vec, typename T>
void matrixMultRSS(const Vec &a, const Vec &b, vector<T> &temp3,
				   size_t rows, size_t common_dim, size_t columns,
				   size_t transpose_a, size_t transpose_b)
{
#if (!USING_EIGEN)
	/********************************* Triple For Loop *********************************/
	Vec triple_a(rows * common_dim), triple_b(common_dim * columns);

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < common_dim; ++j)
		{
			if (transpose_a)
				triple_a[i * common_dim + j] = a[j * rows + i];
			else
				triple_a[i * common_dim + j] = a[i * common_dim + j];
		}
	}

	for (size_t i = 0; i < common_dim; ++i)
	{
		for (size_t j = 0; j < columns; ++j)
		{
			if (transpose_b)
				triple_b[i * columns + j] = b[j * common_dim + i];
			else
				triple_b[i * columns + j] = b[i * columns + j];
		}
	}

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < columns; ++j)
		{
			temp3[i * columns + j] = 0;
			for (int k = 0; k < common_dim; ++k)
			{
				temp3[i * columns + j] += triple_a[i * common_dim + k].first * triple_b[k * columns + j].first +
										  triple_a[i * common_dim + k].first * triple_b[k * columns + j].second +
										  triple_a[i * common_dim + k].second * triple_b[k * columns + j].first;
			}
		}
	}
/********************************* Triple For Loop *********************************/
#endif
#if (USING_EIGEN)
	/********************************* WITH EIGEN Mat-Mul *********************************/
	eigenMatrix eigen_a(rows, common_dim), eigen_b(common_dim, columns), eigen_c(rows, columns);

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < common_dim; ++j)
		{
			if (transpose_a)
			{
				eigen_a.m_share[0](i, j) = a[j * rows + i].first;
				eigen_a.m_share[1](i, j) = a[j * rows + i].second;
			}
			else
			{
				eigen_a.m_share[0](i, j) = a[i * common_dim + j].first;
				eigen_a.m_share[1](i, j) = a[i * common_dim + j].second;
			}
		}
	}

	for (size_t i = 0; i < common_dim; ++i)
	{
		for (size_t j = 0; j < columns; ++j)
		{
			if (transpose_b)
			{
				eigen_b.m_share[0](i, j) = b[j * common_dim + i].first;
				eigen_b.m_share[1](i, j) = b[j * common_dim + i].second;
			}
			else
			{
				eigen_b.m_share[0](i, j) = b[i * columns + j].first;
				eigen_b.m_share[1](i, j) = b[i * columns + j].second;
			}
		}
	}

	eigen_c = eigen_a * eigen_b;

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			temp3[i * columns + j] = eigen_c.m_share[0](i, j);
/********************************* WITH EIGEN Mat-Mul *********************************/
#endif
}

template <typename T>
T dividePlain(T a, int b);
template <typename T>
T dividePlain(T a, int b)
{
	assert((b != 0) && "Cannot divide by 0");

	if (std::is_same<T, lowBit>::value)
		return static_cast<myType>(static_cast<int32_t>(a) / static_cast<int32_t>(b));
	if (std::is_same<T, highBit>::value)
		return static_cast<myType>(static_cast<int64_t>(a) / static_cast<int64_t>(b));
}

template <typename T>
void dividePlain(vector<T> &vec, int divisor);
template <typename T>
void dividePlain(vector<T> &vec, int divisor)
{
	assert((divisor != 0) && "Cannot divide by 0");

	if (std::is_same<T, lowBit>::value)
		for (int i = 0; i < vec.size(); ++i)
			vec[i] = (T)((double)((int32_t)vec[i]) / (double)((int32_t)divisor));

	if (std::is_same<T, highBit>::value)
		for (int i = 0; i < vec.size(); ++i)
			vec[i] = (T)((double)((int64_t)vec[i]) / (double)((int64_t)divisor));
}

size_t nextParty(size_t party);
size_t prevParty(size_t party);

template <typename Vec, typename T>
void merge2Vec(Vec &a, const vector<T> v1, const vector<T> v2, size_t size)
{
	assert(a.size() == v1.size() && v1.size() == v2.size() && v2.size() == size);

	for (size_t i = 0; i < size; i++)
	{
		a[i] = make_pair(v1[i], v2[i]);
	}
}

inline smallType getMSB(lowBit a)
{
	return ((smallType)((a >> (BIT_SIZE_LOW - 1)) & 1));
}

inline smallType getMSB(highBit a)
{
	return ((smallType)((a >> (BIT_SIZE_HIGH - 1)) & 1));
}

inline RSSSmallType addModPrime(RSSSmallType a, RSSSmallType b)
{
	RSSSmallType ret;
	ret.first = additionModPrime[a.first][b.first];
	ret.second = additionModPrime[a.second][b.second];
	return ret;
}

inline smallType subModPrime(smallType a, smallType b)
{
	return subtractModPrime[a][b];
}

inline RSSSmallType subConstModPrime(RSSSmallType a, const smallType r)
{
	RSSSmallType ret = a;
	switch (partyNum)
	{
	case PARTY_A:
		ret.first = subtractModPrime[a.first][r];
		break;
	case PARTY_C:
		ret.second = subtractModPrime[a.second][r];
		break;
	}
	return ret;
}

inline RSSSmallType XORPublicModPrime(RSSSmallType a, bool r)
{
	RSSSmallType ret;
	if (r == 0)
		ret = a;
	else
	{
		switch (partyNum)
		{
		case PARTY_A:
			ret.first = subtractModPrime[1][a.first];
			ret.second = subtractModPrime[0][a.second];
			break;
		case PARTY_B:
			ret.first = subtractModPrime[0][a.first];
			ret.second = subtractModPrime[0][a.second];
			break;
		case PARTY_C:
			ret.first = subtractModPrime[0][a.first];
			ret.second = subtractModPrime[1][a.second];
			break;
		}
	}
	return ret;
}

inline smallType wrapAround(lowBit a, lowBit b)
{
	return (a > ((lowBit)-1) - b);
}

inline smallType wrapAround(highBit a, highBit b)
{
	return (a > ((highBit)-1) - b);
}

inline smallType wrap3(lowBit a, lowBit b, lowBit c)
{
	lowBit temp = a + b;
	if (wrapAround(a, b))
		return 1 - wrapAround(temp, c);
	else
		return wrapAround(temp, c);
}

inline smallType wrap3(highBit a, highBit b, highBit c)
{
	highBit temp = a + b;
	if (wrapAround(a, b))
		return 1 - wrapAround(temp, c);
	else
		return wrapAround(temp, c);
}

template <typename T>
void wrapAround(const vector<T> &a, const vector<T> &b,
				vector<smallType> &c, size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = wrapAround(a[i], b[i]);
}

template <typename VEC, typename T>
void wrap3(const VEC &a, const vector<T> &b,
		   vector<smallType> &c, size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = wrap3(a[i].first, a[i].second, b[i]);
}

template <typename VEC>
void multiplyByScalar(const VEC &a, size_t scalar, VEC &b)
{
	size_t size = a.size();
	for (int i = 0; i < size; ++i)
	{
		b[i].first = a[i].first * scalar;
		b[i].second = a[i].second * scalar;
	}
}
// void transposeVector(const RSSVectorMyType &a, RSSVectorMyType &b, size_t rows, size_t columns);
template <typename VEC>
void zeroPad(const VEC &a, VEC &b,
			 size_t iw, size_t ih, size_t P, size_t Din, size_t B)
{
	size_t size_B = (iw + 2 * P) * (ih + 2 * P) * Din;
	size_t size_Din = (iw + 2 * P) * (ih + 2 * P);
	size_t size_w = (iw + 2 * P);

	for (size_t i = 0; i < B; ++i)
		for (size_t j = 0; j < Din; ++j)
			for (size_t k = 0; k < ih; ++k)
				for (size_t l = 0; l < iw; ++l)
				{
					b[i * size_B + j * size_Din + (k + P) * size_w + l + P] = a[i * Din * iw * ih + j * iw * ih + k * iw + l];
				}
}
// void convToMult(const RSSVectorMyType &vec1, RSSVectorMyType &vec2,
// 				size_t iw, size_t ih, size_t f, size_t Din, size_t S, size_t B);

/*********************** TEMPLATE FUNCTIONS ***********************/

template <typename T, typename U>
std::pair<T, U> operator+(const std::pair<T, U> &l, const std::pair<T, U> &r)
{
	return {l.first + r.first, l.second + r.second};
}

template <typename T, typename U>
std::pair<T, U> operator-(const std::pair<T, U> &l, const std::pair<T, U> &r)
{
	return {l.first - r.first, l.second - r.second};
}

template <typename T, typename U>
std::pair<T, U> operator^(const std::pair<T, U> &l, const std::pair<T, U> &r)
{
	return {l.first ^ r.first, l.second ^ r.second};
}

template <typename T>
T operator*(const std::pair<T, T> &l, const std::pair<T, T> &r)
{
	return {l.first * r.first + l.second * r.first + l.first * r.second};
}

template <typename T, typename U>
std::pair<U, U> operator*(const T a, const std::pair<U, U> &r)
{
	return {a * r.first, a * r.second};
}

template <typename T>
std::pair<T, T> operator<<(const std::pair<T, T> &l, const int shift)
{
	return {l.first << shift, l.second << shift};
}

template <typename T>
void addVectors(const vector<T> &a, const vector<T> &b, vector<T> &c, size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = a[i] + b[i];
}

template <typename T>
void subtractVectors(const vector<T> &a, const vector<T> &b, vector<T> &c, size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = a[i] - b[i];
}

template <typename T>
void copyVectors(const vector<T> &a, vector<T> &b, size_t size)
{
	for (size_t i = 0; i < size; ++i)
		b[i] = a[i];
}

#include <chrono>
#include <utility>

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

template <typename F, typename... Args>
double funcTime(F func, Args &&...args)
{
	TimeVar t1 = timeNow();
	func(std::forward<Args>(args)...);
	return duration(timeNow() - t1);
}

#endif
