
#pragma once
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <tuple>
// #include "BooleanFunc.h"
using namespace std;

// From cpp. Since the template functions have to be implemented here
#include "Precompute.h"
#include <thread>

using namespace std;
extern Precompute *PrecomputeObject;
extern string SECURITY_TYPE;

extern void start_time();
extern void start_communication();
extern void end_time(string str);
extern void end_communication(string str);

void funcTruncatePublic(RSSVectorMyType &a, size_t divisor, size_t size);

template <typename Vec, typename T>
void funcGetShares(Vec &a, const vector<T> &data);
template <typename Vec, typename T>
void funcGetShares(Vec &a, const vector<T> &data)
{
	size_t size = a.size();

	if (partyNum == PARTY_A)
	{
		for (int i = 0; i < size; ++i)
		{
			a[i].first = data[i];
			a[i].second = 0;
		}
	}
	else if (partyNum == PARTY_B)
	{
		for (int i = 0; i < size; ++i)
		{
			a[i].first = 0;
			a[i].second = 0;
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (int i = 0; i < size; ++i)
		{
			a[i].first = 0;
			a[i].second = data[i];
		}
	}
}

void funcBoolPartySS(RSSVectorBoolType &result, const vector<bool> &data, size_t size, int shareParty);

/**
 * @brief only shareParty know the plain of data
 * shareParty generate ss of data and send to other parties
 *
 * @tparam Vec
 * @tparam T
 * @param a
 * @param data
 * @param size
 */
template <typename Vec, typename T>
void funcPartySS(Vec &a, const vector<T> &data, const size_t size, const int shareParty)
{
	assert(a.size() == size && a.size() == data.size() && "a.size mismatch for PartyShare function");
	PrecomputeObject->getZeroShareRand<Vec, T>(a, size, shareParty);
	// cout << "share party " << shareParty << " partyNum" << partyNum << endl;
	if (partyNum == shareParty)
	{
		vector<T> a1_plus_data(size);
		for (size_t i = 0; i < size; i++)
		{
			T temp = data[i] + a[i].first;
			a[i].first = temp;
			a1_plus_data[i] = temp;
		}

		// send a1+x to prevparty
		sendVector<T>(a1_plus_data, prevParty(partyNum), size);
		// thread sender(sendVector<T>, ref(a1_plus_data), prevParty(partyNum), size);
		// sender.join();
	}
	else if (partyNum == prevParty(shareParty))
	{

		vector<T> a1_plus_data(size);
		receiveVector<T>(a1_plus_data, shareParty, size); // receive a1+x

		for (size_t i = 0; i < size; i++)
		{
			a[i].second = a1_plus_data[i];
		}
		// merge2Vec<Vec, T>(a, a3, a1_plus_data, size);
	}
}

/**
 * @brief generate ss of data and send to other parties
 *
 * @tparam Vec
 * @tparam T
 * @param a secret share of plain data
 * @param data plain data
 * @param size
 */
template <typename Vec, typename T>
void funcShareSender(Vec &a, const vector<T> &data, const size_t size)
{
	assert(a.size() == size && "a.size mismatch for share function");

	PrecomputeObject->getZeroShareSender<Vec, T>(a, size);

	vector<T> a1_plus_data(size);
	for (size_t i = 0; i < size; i++)
	{
		T temp = data[i] + a[i].first;
		a[i].first = temp;
		a1_plus_data[i] = temp;
	}

	// send a1+x to prevparty
	thread sender(sendVector<T>, ref(a1_plus_data), prevParty(partyNum), size);
	sender.join();
}

/**
 * @brief receive share from shareParty
 *
 * @tparam Vec
 * @tparam T
 * @param a
 * @param size
 * @param shareParty
 */
template <typename Vec, typename T>
void funcShareReceiver(Vec &a, const size_t size, const int shareParty)
{
	assert(a.size() == size && "a.size mismatch for share function");

	if (partyNum == prevParty(shareParty))
	{
		vector<T> a3(size);
		PrecomputeObject->getZeroSharePrev<T>(a3, size);

		vector<T> a1_plus_data(size);
		thread receiver(receiveVector<T>, ref(a1_plus_data), shareParty, size); // receive a1+x
		receiver.join();

		merge2Vec<Vec, T>(a, a3, a1_plus_data, size);
	}
	else
	{
		PrecomputeObject->getZeroShareReceiver<Vec, T>(a, size);
	}
}

void funcGetShares(RSSVectorSmallType &a, const vector<smallType> &data);
void funcReconstructBit(const RSSVectorSmallType &a, vector<smallType> &b, size_t size, string str, bool print);

template <typename VEC, typename T>
void funcReconstruct(const VEC &a, vector<T> &b, size_t size, string str, bool print);
template <typename VEC, typename T>
void funcReconstruct(const VEC &a, vector<T> &b, size_t size, string str, bool print)
{
	log_print("Reconst: RSSMyType, myType");
	assert(a.size() == size && "a.size mismatch for reconstruct function");

	if (SECURITY_TYPE.compare("Semi-honest") == 0)
	{
		vector<T> a_next(size), a_prev(size);
		for (int i = 0; i < size; ++i)
		{
			a_prev[i] = 0;
			a_next[i] = a[i].first;
			b[i] = a[i].first;
			if (std::is_same<T, smallType>::value)
				b[i] = additionModPrime[b[i]][a[i].second];
			else
				b[i] = b[i] + a[i].second;
		}

		thread *threads = new thread[2];

		threads[0] = thread(sendVector<T>, ref(a_next), nextParty(partyNum), size);
		threads[1] = thread(receiveVector<T>, ref(a_prev), prevParty(partyNum), size);

		for (int i = 0; i < 2; i++)
			threads[i].join();

		delete[] threads;

		for (int i = 0; i < size; ++i)
		{
			if (std::is_same<T, smallType>::value)
				b[i] = additionModPrime[b[i]][a_prev[i]];
			else
				b[i] = b[i] + a_prev[i];
		}

		if (print)
		{
			printVectorReal(b, str, size);
			// std::cout << str << endl;
			// for (int i = 0; i < size; ++i)
			// 	print_linear(b[i], "SIGNED");
			// std::cout << std::endl;
		}
	}

	if (SECURITY_TYPE.compare("Malicious") == 0)
	{
		vector<smallType> a_next_send(size), a_prev_send(size), a_next_recv(size), a_prev_recv(size);
		for (int i = 0; i < size; ++i)
		{
			a_prev_send[i] = a[i].second;
			a_next_send[i] = a[i].first;
			b[i] = a[i].first;
			if (!std::is_same<T, smallType>::value)
				b[i] = b[i] + a[i].second;
		}

		thread *threads = new thread[4];
		threads[0] = thread(sendVector<smallType>, ref(a_next_send), nextParty(partyNum), size);
		threads[1] = thread(sendVector<smallType>, ref(a_prev_send), prevParty(partyNum), size);
		threads[2] = thread(receiveVector<smallType>, ref(a_next_recv), nextParty(partyNum), size);
		threads[3] = thread(receiveVector<smallType>, ref(a_prev_recv), prevParty(partyNum), size);
		for (int i = 0; i < 4; i++)
			threads[i].join();
		delete[] threads;

		for (int i = 0; i < size; ++i)
		{
			if (a_next_recv[i] != a_prev_recv[i])
			{
				// error("Malicious behaviour detected");
			}
			if (!std::is_same<T, smallType>::value)
				b[i] = b[i] + a_prev_recv[i];
		}

		if (print)
		{
			std::cout << str << ": \t\t";
			for (int i = 0; i < size; ++i)
				print_linear(b[i], "SIGNED");
			std::cout << std::endl;
		}
	}
}

void funcReconstruct(const RSSVectorSmallType &a, vector<smallType> &b, size_t size, string str, bool print);

template <typename T>
void funcReconstruct3out3(const vector<T> &a, vector<T> &b, size_t size, string str, bool print);
// Asymmetric protocol for semi-honest setting.
template <typename T>
void funcReconstruct3out3(const vector<T> &a, vector<T> &b, size_t size, string str, bool print)
{
	log_print("Reconst: myType, myType");
	assert(a.size() == size && "a.size mismatch for reconstruct function");

	vector<T> temp_A(size, 0), temp_B(size, 0);

	if (partyNum == PARTY_A or partyNum == PARTY_B)
		sendVector<T>(a, PARTY_C, size);

	if (partyNum == PARTY_C)
	{
		receiveVector<T>(temp_A, PARTY_A, size);
		receiveVector<T>(temp_B, PARTY_B, size);
		addVectors<T>(temp_A, a, temp_A, size);
		addVectors<T>(temp_B, temp_A, b, size);
		sendVector<T>(b, PARTY_A, size);
		sendVector<T>(b, PARTY_B, size);
	}

	if (partyNum == PARTY_A or partyNum == PARTY_B)
		receiveVector<T>(b, PARTY_C, size);

	if (print)
	{
		std::cout << str << ": \t\t";
		for (int i = 0; i < size; ++i)
			print_linear(b[i], "SIGNED");
		std::cout << std::endl;
	}

	if (SECURITY_TYPE.compare("Semi-honest") == 0)
	{
	}

	if (SECURITY_TYPE.compare("Malicious") == 0)
	{
	}
}

template <typename VEC>
void funcTruncate(VEC &a, size_t power, size_t size)
{
	log_print("funcTruncate");

	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, highBit, lowBit>::type computeType;

	VEC r(size), rPrime(size);
	vector<computeType> reconst(size);
	PrecomputeObject->getDividedShares(r, rPrime, (1 << power), size);
	for (int i = 0; i < size; ++i)
		a[i] = a[i] - rPrime[i];

	funcReconstruct(a, reconst, size, "Truncate reconst", false);
	dividePlain(reconst, (1l << power));
	if (partyNum == PARTY_A)
	{
		for (int i = 0; i < size; ++i)
		{
			a[i].first = r[i].first + reconst[i];
			a[i].second = r[i].second;
		}
	}

	if (partyNum == PARTY_B)
	{
		for (int i = 0; i < size; ++i)
		{
			a[i].first = r[i].first;
			a[i].second = r[i].second;
		}
	}

	if (partyNum == PARTY_C)
	{
		for (int i = 0; i < size; ++i)
		{
			a[i].first = r[i].first;
			a[i].second = r[i].second + reconst[i];
		}
	}
}

template <typename vec, typename T>
void funcMulPlain(vec &result, vec &input, vector<T> c, size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		result[i].first = input[i].first * c[i];
		result[i].second = input[i].second * c[i];
	}
}

template <typename vec, typename T>
void funcMulConst(vec &result, vec &input, T c, size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		result[i].first = input[i].first * c;
		result[i].second = input[i].second * c;
	}
}

template <typename Vec>
void funcMatMul(const Vec &a, const Vec &b, Vec &c,
				size_t rows, size_t common_dim, size_t columns,
				size_t transpose_a, size_t transpose_b, size_t truncation);

template <typename T>
void funcDotProduct(const T &a, const T &b,
					T &c, size_t size, bool truncation, size_t precision);
void funcDotProduct(const RSSVectorSmallType &a, const RSSVectorSmallType &b,
					RSSVectorSmallType &c, size_t size);

inline void funcCheckMaliciousDotProd(const RSSVectorSmallType &a, const RSSVectorSmallType &b, const RSSVectorSmallType &c,
									  const vector<smallType> &temp, size_t size)
{
	RSSVectorSmallType x(size), y(size), z(size);
	PrecomputeObject->getTriplets(x, y, z, size);

	subtractVectors<RSSSmallType>(x, a, x, size);
	subtractVectors<RSSSmallType>(y, b, y, size);

	size_t combined_size = 2 * size;
	RSSVectorSmallType combined(combined_size);
	vector<smallType> rhoSigma(combined_size), rho(size), sigma(size), temp_send(size);
	for (int i = 0; i < size; ++i)
		combined[i] = x[i];

	for (int i = size; i < combined_size; ++i)
		combined[i] = y[i - size];

	funcReconstruct(combined, rhoSigma, combined_size, "rhoSigma", false);

	for (int i = 0; i < size; ++i)
		rho[i] = rhoSigma[i];

	for (int i = size; i < combined_size; ++i)
		sigma[i - size] = rhoSigma[i];

	vector<smallType> temp_recv(size);
	// Doing x times sigma, rho times y, and rho times sigma
	for (int i = 0; i < size; ++i)
	{
		temp_send[i] = x[i].first + sigma[i];
		temp_send[i] = rho[i] + y[i].first;
		temp_send[i] = rho[i] + sigma[i];
	}

	thread *threads = new thread[2];
	threads[0] = thread(sendVector<smallType>, ref(temp_send), nextParty(partyNum), size);
	threads[1] = thread(receiveVector<smallType>, ref(temp_recv), prevParty(partyNum), size);
	for (int i = 0; i < 2; i++)
		threads[i].join();
	delete[] threads;

	for (int i = 0; i < size; ++i)
		if (temp[i] == temp_recv[i])
		{
			// Do the abort thing
		}
}

// Multiply index 2i, 2i+1 of the first vector into the second one. The second vector is half the size.
inline void funcMultiplyNeighbours(const RSSVectorSmallType &c_1, RSSVectorSmallType &c_2, size_t size)
{
	assert(size % 2 == 0 && "Size should be 'half'able");
	vector<smallType> temp3(size / 2, 0), recv(size / 2, 0);
	for (int i = 0; i < size / 2; ++i)
	{
		temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[c_1[2 * i].first][c_1[2 * i + 1].first]];
		temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[c_1[2 * i].first][c_1[2 * i + 1].second]];
		temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[c_1[2 * i].second][c_1[2 * i + 1].first]];
	}

	// Add random shares of 0 locally
	thread *threads = new thread[2];

	threads[0] = thread(sendVector<smallType>, ref(temp3), nextParty(partyNum), size / 2);
	threads[1] = thread(receiveVector<smallType>, ref(recv), prevParty(partyNum), size / 2);

	for (int i = 0; i < 2; i++)
		threads[i].join();
	delete[] threads;

	for (int i = 0; i < size / 2; ++i)
	{
		c_2[i].first = temp3[i];
		c_2[i].second = recv[i];
	}

	RSSVectorSmallType temp_a(size / 2), temp_b(size / 2), temp_c(size / 2);
	if (SECURITY_TYPE.compare("Malicious") == 0)
		funcCheckMaliciousDotProd(temp_a, temp_b, temp_c, temp3, size / 2);
}

inline void funcCrunchMultiply(const RSSVectorSmallType &c, vector<smallType> &betaPrime, size_t size, int bits_size)
{
	size_t sizeLong = size * bits_size;
	RSSVectorSmallType c_0(sizeLong / 2, make_pair(0, 0)), c_1(sizeLong / 4, make_pair(0, 0)),
		c_2(sizeLong / 8, make_pair(0, 0)), c_3(sizeLong / 16, make_pair(0, 0)),
		c_4(sizeLong / 32, make_pair(0, 0));
	RSSVectorSmallType c_5(sizeLong / 64, make_pair(0, 0));

	vector<smallType> reconst(size, 0);

	funcMultiplyNeighbours(c, c_0, sizeLong);
	funcMultiplyNeighbours(c_0, c_1, sizeLong / 2);
	funcMultiplyNeighbours(c_1, c_2, sizeLong / 4);
	funcMultiplyNeighbours(c_2, c_3, sizeLong / 8);
	funcMultiplyNeighbours(c_3, c_4, sizeLong / 16);
	if (bits_size == 64)
		funcMultiplyNeighbours(c_4, c_5, sizeLong / 32);

	vector<smallType> a_next(size), a_prev(size);
	if (bits_size == 64)
		for (int i = 0; i < size; ++i)
		{
			a_prev[i] = 0;
			a_next[i] = c_5[i].first;
			reconst[i] = c_5[i].first;
			reconst[i] = additionModPrime[reconst[i]][c_5[i].second];
		}
	else if (bits_size == 32)
		for (int i = 0; i < size; ++i)
		{
			a_prev[i] = 0;
			a_next[i] = c_4[i].first;
			reconst[i] = c_4[i].first;
			reconst[i] = additionModPrime[reconst[i]][c_4[i].second];
		}

	thread *threads = new thread[2];

	threads[0] = thread(sendVector<smallType>, ref(a_next), nextParty(partyNum), size);
	threads[1] = thread(receiveVector<smallType>, ref(a_prev), prevParty(partyNum), size);
	for (int i = 0; i < 2; i++)
		threads[i].join();
	delete[] threads;

	for (int i = 0; i < size; ++i)
		reconst[i] = additionModPrime[reconst[i]][a_prev[i]];

	for (int i = 0; i < size; ++i)
	{
		if (reconst[i] == 0)
			betaPrime[i] = 1;
	}
}

// Thread function for parallel private compare
template <typename T>
void parallelFirst(smallType *temp3, const RSSSmallType *beta, const T *r,
				   const RSSSmallType *share_m, size_t start, size_t end, int t, int bits_size)
{
	size_t index3, index2;
	smallType bit_r;
	RSSSmallType twoBetaMinusOne, diff;

	for (int index2 = start; index2 < end; ++index2)
	{
		// Computing 2Beta-1
		twoBetaMinusOne = subConstModPrime(beta[index2], 1);
		twoBetaMinusOne = addModPrime(twoBetaMinusOne, beta[index2]);

		for (size_t k = 0; k < bits_size; ++k)
		{
			index3 = index2 * bits_size + k;
			bit_r = (smallType)((r[index2] >> (bits_size - 1 - k)) & 1);
			diff = share_m[index3];

			if (bit_r == 1)
				diff = subConstModPrime(diff, 1);

			// Dot Product
			temp3[index3] = multiplicationModPrime[diff.first][twoBetaMinusOne.first];
			temp3[index3] = additionModPrime[temp3[index3]][multiplicationModPrime[diff.first][twoBetaMinusOne.second]];
			temp3[index3] = additionModPrime[temp3[index3]][multiplicationModPrime[diff.second][twoBetaMinusOne.first]];
		}
	}
}

// void parallelFirst(smallType* temp3, const RSSSmallType* beta, const lowBit* r,
// 					const RSSSmallType* share_m, size_t start, size_t end, int t, int bits_size)
// {
// 	size_t index3, index2;
// 	smallType bit_r;
// 	RSSSmallType twoBetaMinusOne, diff;

// 	for (int index2 = start; index2 < end; ++index2)
// 	{
// 		//Computing 2Beta-1
// 		twoBetaMinusOne = subConstModPrime(beta[index2], 1);
// 		twoBetaMinusOne = addModPrime(twoBetaMinusOne, beta[index2]);

// 		for (size_t k = 0; k < bits_size; ++k)
// 		{
// 			index3 = index2*bits_size + k;
// 			bit_r = (smallType)((r[index2] >> (bits_size-1-k)) & 1);
// 			diff = share_m[index3];

// 			if (bit_r == 1)
// 				diff = subConstModPrime(diff, 1);

// 			//Dot Product
// 			temp3[index3] = multiplicationModPrime[diff.first][twoBetaMinusOne.first];
// 			temp3[index3] = additionModPrime[temp3[index3]][multiplicationModPrime[diff.first][twoBetaMinusOne.second]];
// 			temp3[index3] = additionModPrime[temp3[index3]][multiplicationModPrime[diff.second][twoBetaMinusOne.first]];
// 		}
// 	}
// }

template <typename T>
void parallelSecond(RSSSmallType *c, const smallType *temp3, const smallType *recv, const T *r,
					const RSSSmallType *share_m, size_t start, size_t end, int t, int bits_size)
{
	size_t index3, index2;
	smallType bit_r;
	RSSSmallType a, tempM, tempN, xMinusR;

	if (partyNum == PARTY_A)
	{
		for (int index2 = start; index2 < end; ++index2)
		{
			a = make_pair(0, 0);
			for (size_t k = 0; k < bits_size; ++k)
			{
				index3 = index2 * bits_size + k;
				// Complete Dot Product
				xMinusR.first = temp3[index3];
				xMinusR.second = recv[index3];

				// Resume rest of the loop
				c[index3] = a;
				tempM = share_m[index3];
				bit_r = (smallType)((r[index2] >> (bits_size - 1 - k)) & 1);

				tempN = XORPublicModPrime(tempM, bit_r);
				a = addModPrime(a, tempN);

				c[index3].first = additionModPrime[c[index3].first][xMinusR.first];
				c[index3].first = additionModPrime[c[index3].first][1];
				c[index3].second = additionModPrime[c[index3].second][xMinusR.second];
			}
		}
	}

	if (partyNum == PARTY_B)
	{
		for (int index2 = start; index2 < end; ++index2)
		{
			a = make_pair(0, 0);
			for (size_t k = 0; k < bits_size; ++k)
			{
				index3 = index2 * bits_size + k;
				// Complete Dot Product
				xMinusR.first = temp3[index3];
				xMinusR.second = recv[index3];

				// Resume rest of the loop
				c[index3] = a;
				tempM = share_m[index3];
				bit_r = (smallType)((r[index2] >> (bits_size - 1 - k)) & 1);

				tempN = XORPublicModPrime(tempM, bit_r);
				a = addModPrime(a, tempN);

				c[index3].first = additionModPrime[c[index3].first][xMinusR.first];
				c[index3].second = additionModPrime[c[index3].second][xMinusR.second];
			}
		}
	}

	if (partyNum == PARTY_C)
	{
		for (int index2 = start; index2 < end; ++index2)
		{
			a = make_pair(0, 0);
			for (size_t k = 0; k < bits_size; ++k)
			{
				index3 = index2 * bits_size + k;
				// Complete Dot Product
				xMinusR.first = temp3[index3];
				xMinusR.second = recv[index3];

				// Resume rest of the loop
				c[index3] = a;
				tempM = share_m[index3];
				bit_r = (smallType)((r[index2] >> (bits_size - 1 - k)) & 1);

				tempN = XORPublicModPrime(tempM, bit_r);
				a = addModPrime(a, tempN);

				c[index3].first = additionModPrime[c[index3].first][xMinusR.first];
				c[index3].second = additionModPrime[c[index3].second][xMinusR.second];
				c[index3].second = additionModPrime[c[index3].second][1];
			}
		}
	}
}

template <typename T>
void funcPrivateCompare(const RSSVectorSmallType &share_m, const vector<T> &r,
						const RSSVectorSmallType &beta, vector<smallType> &betaPrime,
						size_t size)
{
	log_print("funcPrivateCompare");
	// cout << "wrap input: " <<  typeid(r).name() << endl;

	int bits_size = BIT_SIZE;
	if (std::is_same<T, highBit>::value)
	{
		bits_size = BIT_SIZE_HIGH;
	}
	else
	{
		bits_size = BIT_SIZE_LOW;
	}

	assert(share_m.size() == size * bits_size && "Input error share_m");
	assert(r.size() == size && "Input error r");
	assert(beta.size() == size && "Input error beta");

	size_t sizeLong = size * bits_size;
	size_t index3, index2;
	RSSVectorSmallType c(sizeLong), diff(sizeLong), twoBetaMinusOne(sizeLong), xMinusR(sizeLong);
	RSSSmallType a, tempM, tempN;
	smallType bit_r;

	// Computing x[i] - r[i]
	if (PARALLEL)
	{
		assert(NO_CORES > 2 && "Need at least 2 cores for threads variable abuse");
		vector<smallType> temp3(sizeLong, 0), recv(sizeLong, 0);

		// First part of parallel execution
		thread *threads = new thread[NO_CORES];
		int chunksize = size / NO_CORES;

		for (int i = 0; i < NO_CORES; i++)
		{
			int start = i * chunksize;
			int end = (i + 1) * chunksize;
			if (i == NO_CORES - 1)
				end = size;

			threads[i] = thread(parallelFirst<T>, temp3.data(), beta.data(), r.data(),
								share_m.data(), start, end, i, bits_size);
		}
		for (int i = 0; i < NO_CORES; i++)
			threads[i].join();

		//"Single" threaded execution
		threads[0] = thread(sendVector<smallType>, ref(temp3), prevParty(partyNum), size);
		threads[1] = thread(receiveVector<smallType>, ref(recv), nextParty(partyNum), size);

		for (int i = 0; i < 2; i++)
			threads[i].join();

		// Parallel execution resumes
		for (int i = 0; i < NO_CORES; i++)
		{
			int start = i * chunksize;
			int end = (i + 1) * chunksize;
			if (i == NO_CORES - 1)
				end = size;

			threads[i] = thread(parallelSecond<T>, c.data(), temp3.data(), recv.data(),
								r.data(), share_m.data(), start, end, i, bits_size);
		}

		for (int i = 0; i < NO_CORES; i++)
			threads[i].join();
		delete[] threads;
	}
	else
	{
		for (int index2 = 0; index2 < size; ++index2)
		{
			// Computing 2Beta-1
			twoBetaMinusOne[index2 * bits_size] = subConstModPrime(beta[index2], 1);
			twoBetaMinusOne[index2 * bits_size] = addModPrime(twoBetaMinusOne[index2 * bits_size], beta[index2]);

			for (size_t k = 0; k < bits_size; ++k)
			{
				index3 = index2 * bits_size + k;
				twoBetaMinusOne[index3] = twoBetaMinusOne[index2 * bits_size];

				bit_r = (smallType)((r[index2] >> (bits_size - 1 - k)) & 1);
				diff[index3] = share_m[index3];

				if (bit_r == 1)
					diff[index3] = subConstModPrime(diff[index3], 1);
			}
		}

		//(-1)^beta * x[i] - r[i]
		funcDotProduct(diff, twoBetaMinusOne, xMinusR, sizeLong);

		for (int index2 = 0; index2 < size; ++index2)
		{
			a = make_pair(0, 0);
			for (size_t k = 0; k < bits_size; ++k)
			{
				index3 = index2 * bits_size + k;
				c[index3] = a;
				tempM = share_m[index3];

				bit_r = (smallType)((r[index2] >> (bits_size - 1 - k)) & 1);

				tempN = XORPublicModPrime(tempM, bit_r);
				a = addModPrime(a, tempN);

				if (partyNum == PARTY_A)
				{
					c[index3].first = additionModPrime[c[index3].first][xMinusR[index3].first];
					c[index3].first = additionModPrime[c[index3].first][1];
					c[index3].second = additionModPrime[c[index3].second][xMinusR[index3].second];
				}
				else if (partyNum == PARTY_B)
				{
					c[index3].first = additionModPrime[c[index3].first][xMinusR[index3].first];
					c[index3].second = additionModPrime[c[index3].second][xMinusR[index3].second];
				}
				else if (partyNum == PARTY_C)
				{
					c[index3].first = additionModPrime[c[index3].first][xMinusR[index3].first];
					c[index3].second = additionModPrime[c[index3].second][xMinusR[index3].second];
					c[index3].second = additionModPrime[c[index3].second][1];
				}
			}
		}
	}

	RSSVectorSmallType temp_a(sizeLong / 2), temp_b(sizeLong / 2), temp_c(sizeLong / 2);
	vector<smallType> temp_d(sizeLong / 2);
	if (SECURITY_TYPE.compare("Malicious") == 0)
		funcCheckMaliciousDotProd(temp_a, temp_b, temp_c, temp_d, sizeLong / 2);

	// TODO 7 rounds of multiplication
	//  cout << "CM: \t\t" << funcTime(funcCrunchMultiply, c, betaPrime, size, dim) << endl;
	funcCrunchMultiply(c, betaPrime, size, bits_size);
}

template <typename VEC>
void funcWrap(const VEC &a, RSSVectorSmallType &theta, size_t size)
{
	log_print("funcWrap");
	// cout << "wrap input: " <<  typeid(a).name() << endl;

	int bits_size = BIT_SIZE;
	if (std::is_same<VEC, RSSVectorHighType>::value)
	{
		bits_size = BIT_SIZE_HIGH;
	}
	else
	{
		bits_size = BIT_SIZE_LOW;
	}
	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, highBit, lowBit>::type computeType;
	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, RSSHighType, RSSLowType>::type RSScomputeType;

	size_t sizeLong = size * bits_size;
	VEC x(size), r(size);
	RSSVectorSmallType shares_r(sizeLong), alpha(size), beta(size), eta(size);
	vector<smallType> delta(size), etaPrime(size);
	vector<computeType> reconst_x(size);

	PrecomputeObject->getShareConvertObjects(r, shares_r, alpha, size);

	addVectors<RSScomputeType>(a, r, x, size);
	for (int i = 0; i < size; ++i)
	{
		beta[i].first = wrapAround(a[i].first, r[i].first);
		x[i].first = a[i].first + r[i].first;
		beta[i].second = wrapAround(a[i].second, r[i].second);
		x[i].second = a[i].second + r[i].second;
	}

	vector<computeType> x_next(size), x_prev(size);
	for (int i = 0; i < size; ++i)
	{
		x_prev[i] = 0;
		x_next[i] = x[i].first;
		reconst_x[i] = x[i].first;
		reconst_x[i] = reconst_x[i] + x[i].second;
	}

	thread *threads = new thread[2];
	threads[0] = thread(sendVector<computeType>, ref(x_next), nextParty(partyNum), size);
	threads[1] = thread(receiveVector<computeType>, ref(x_prev), prevParty(partyNum), size);
	for (int i = 0; i < 2; i++)
		threads[i].join();
	delete[] threads;

	for (int i = 0; i < size; ++i)
		reconst_x[i] = reconst_x[i] + x_prev[i];

	wrap3(x, x_prev, delta, size); // All parties have delta
	PrecomputeObject->getRandomBitShares(eta, size);

	// cout << "PC: \t\t" << funcTime(funcPrivateCompare, shares_r, reconst_x, eta, etaPrime, size, bits_size) << endl;
	funcPrivateCompare(shares_r, reconst_x, eta, etaPrime, size);

	if (partyNum == PARTY_A)
	{
		for (int i = 0; i < size; ++i)
		{
			theta[i].first = beta[i].first ^ delta[i] ^ alpha[i].first ^ eta[i].first ^ etaPrime[i];
			theta[i].second = beta[i].second ^ alpha[i].second ^ eta[i].second;
		}
	}
	else if (partyNum == PARTY_B)
	{
		for (int i = 0; i < size; ++i)
		{
			theta[i].first = beta[i].first ^ delta[i] ^ alpha[i].first ^ eta[i].first;
			theta[i].second = beta[i].second ^ alpha[i].second ^ eta[i].second;
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (int i = 0; i < size; ++i)
		{
			theta[i].first = beta[i].first ^ alpha[i].first ^ eta[i].first;
			theta[i].second = beta[i].second ^ delta[i] ^ alpha[i].second ^ eta[i].second ^ etaPrime[i];
		}
	}
}

void funcSelectShares(const RSSVectorMyType &a, const RSSVectorSmallType &b, RSSVectorMyType &selected, size_t size);
void funcSelectBitShares(const RSSVectorSmallType &a0, const RSSVectorSmallType &a1,
						 const RSSVectorSmallType &b, RSSVectorSmallType &answer,
						 size_t rows, size_t columns, size_t loopCounter);

template <typename VEC>
void funcRELUPrime(const VEC &a, RSSVectorSmallType &b, size_t size)
{
	log_print("funcRELUPrime");
	// cout << "ReLUPrme input: " <<  typeid(a).name() << endl;

	VEC twoA(size);
	RSSVectorSmallType theta(size);
	for (int i = 0; i < size; ++i)
		twoA[i] = a[i] << 1;

	// cout << "Wrap: \t\t" << funcTime(funcWrap, twoA, theta, size) << endl;
	funcWrap(twoA, theta, size);

	for (int i = 0; i < size; ++i)
	{
		b[i].first = theta[i].first ^ (getMSB(a[i].first));
		b[i].second = theta[i].second ^ (getMSB(a[i].second));
	}
}

template <typename VEC>
void funcRELU(const VEC &a, RSSVectorSmallType &temp, VEC &b, size_t size)
{
	log_print("funcRELU");
	// cout << "ReLU input: " <<  typeid(a).name() << endl;
	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, highBit, lowBit>::type computeType;

	RSSVectorSmallType c(size), bXORc(size);
	VEC m_c(size);
	vector<smallType> reconst_b(size);

	// cout << "ReLU': \t\t" << funcTime(funcRELUPrime, a, temp, size) << endl;
	funcRELUPrime(a, temp, size);
	PrecomputeObject->getSelectorBitShares(c, m_c, size);

	for (int i = 0; i < size; ++i)
	{
		bXORc[i].first = c[i].first ^ temp[i].first;
		bXORc[i].second = c[i].second ^ temp[i].second;
	}

	funcReconstructBit(bXORc, reconst_b, size, "bXORc", false);
	if (partyNum == PARTY_A)
		for (int i = 0; i < size; ++i)
			if (reconst_b[i] == 0)
			{
				m_c[i].first = (computeType)1 - m_c[i].first;
				m_c[i].second = -m_c[i].second;
			}

	if (partyNum == PARTY_B)
		for (int i = 0; i < size; ++i)
			if (reconst_b[i] == 0)
			{
				m_c[i].first = -m_c[i].first;
				m_c[i].second = -m_c[i].second;
			}

	if (partyNum == PARTY_C)
		for (int i = 0; i < size; ++i)
			if (reconst_b[i] == 0)
			{
				m_c[i].first = -m_c[i].first;
				m_c[i].second = (computeType)1 - m_c[i].second;
			}

	// vector<computeType> reconst_m_c(size);
	// funcReconstruct(m_c, reconst_m_c, size, "m_c", true);
	funcDotProduct(a, m_c, b, size, false, 0);
}

// TODO
template <typename VEC>
void funcPow(const VEC &b, vector<smallType> &alpha, size_t size)
{
	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, highBit, lowBit>::type computeType;
	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, RSSHighType, RSSLowType>::type RSScomputeType;

	int bits_size = BIT_SIZE;
	if (std::is_same<VEC, RSSVectorHighType>::value)
	{
		bits_size = BIT_SIZE_HIGH;
	}
	else
	{
		bits_size = BIT_SIZE_LOW;
	}

	size_t ell = 5;
	if (bits_size == 64)
		ell = 6;

	VEC x(size), d(size), temp(size);
	copyVectors<RSScomputeType>(b, x, size);

	RSSVectorSmallType c(size);
	for (int i = 0; i < size; ++i)
		alpha[i] = 0;

	vector<smallType> r_c(size);
	// TODO vecrorize this, right now only accepts the first argument
	for (int j = ell - 1; j > -1; --j)
	{
		// This is a potential bug for computing over 64-bit ring. 1 << 32+ = 0
		vector<computeType> temp_1(size, (1l << ((1 << j) + (int)alpha[0])));
		funcGetShares(temp, temp_1);
		subtractVectors(x, temp, d, size);
		funcRELUPrime(d, c, size);
		funcReconstructBit(c, r_c, size, "null", false);
		if (r_c[0] == 0)
		{
			for (int i = 0; i < size; ++i)
				alpha[i] += (1 << j);
		}
	}
}

// template <typename VEC>
// void funcDivision(const VEC &a, const VEC &b, VEC &quotient,
// 				  size_t size)
// {
// 	log_print("funcDivision");

// 	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, highBit, lowBit>::type computeType;
// 	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, RSSHighType, RSSLowType>::type RSScomputeType;

// 	// TODO incorporate funcPow
// 	// TODO Scale up and complete this computation with fixed-point precision
// 	vector<smallType> alpha_temp(size);
// 	funcPow(b, alpha_temp, size);

// 	size_t alpha = alpha_temp[0];
// 	size_t precision = alpha + 1;
// 	const computeType constTwoPointNine = ((computeType)(2.9142 * (1 << precision)));
// 	const computeType constOne = ((computeType)(1 * (1 << precision)));

// 	size_t float_precision = FLOAT_PRECISION;
// 	if (std::is_same<VEC, RSSVectorHighType>::value)
// 	{
// 		float_precision = HIGH_PRECISION;
// 	}
// 	else if (std::is_same<VEC, RSSVectorLowType>::value)
// 	{
// 		float_precision = LOW_PRECISION;
// 	}
// 	else
// 	{
// 		cout << "Not supported type" << typeid(a).name() << endl;
// 	}

// 	vector<computeType> data_twoPointNine(size, constTwoPointNine), data_one(size, constOne), reconst(size);
// 	VEC ones(size), twoPointNine(size), twoX(size), w0(size), xw0(size),
// 		epsilon0(size), epsilon1(size), termOne(size), termTwo(size), answer(size);
// 	funcGetShares(twoPointNine, data_twoPointNine);
// 	funcGetShares(ones, data_one);

// 	multiplyByScalar(b, 2, twoX);
// 	subtractVectors<RSScomputeType>(twoPointNine, twoX, w0, size);
// 	funcDotProduct(b, w0, xw0, size, true, precision);
// 	subtractVectors<RSScomputeType>(ones, xw0, epsilon0, size);
// 	if (PRECISE_DIVISION)
// 		funcDotProduct(epsilon0, epsilon0, epsilon1, size, true, precision);
// 	addVectors(ones, epsilon0, termOne, size);
// 	if (PRECISE_DIVISION)
// 		addVectors(ones, epsilon1, termTwo, size);
// 	funcDotProduct(w0, termOne, answer, size, true, precision);
// 	if (PRECISE_DIVISION)
// 		funcDotProduct(answer, termTwo, answer, size, true, precision);

// 	// RSSVectorMyType scaledA(size);
// 	// multiplyByScalar(a, (1 << (alpha + 1)), scaledA);
// 	funcDotProduct(answer, a, quotient, size, true, ((2 * precision - float_precision)));
// }

template <typename VEC>
void funcDivisionByNR(VEC &a, const VEC &b, const VEC &quotient,
					  size_t size);

template <typename VEC>
void funcBatchNorm(const VEC &a, const VEC &b, VEC &quotient,
				   size_t batchSize, size_t B)
{
	log_print("funcBatchNorm");
	// TODO Scale up and complete this computation with higher fixed-point precision
	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, highBit, lowBit>::type computeType;
	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, RSSHighType, RSSLowType>::type RSScomputeType;

	assert(a.size() == batchSize * B && "funcBatchNorm a size incorrect");
	assert(b.size() == B && "funcBatchNorm b size incorrect");
	assert(quotient.size() == batchSize * B && "funcBatchNorm quotient size incorrect");

	vector<smallType> alpha_temp(B);
	funcPow(b, alpha_temp, B);

	// TODO Get alpha from previous computation
	size_t alpha = alpha_temp[0];
	size_t precision = alpha + 1;
	const computeType constTwoPointNine = ((computeType)(2.9142 * (1 << precision)));
	const computeType constOne = ((computeType)(1 * (1 << precision)));

	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<VEC, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<VEC, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(a).name() << endl;
	}

	vector<computeType> data_twoPointNine(B, constTwoPointNine), data_one(B, constOne), reconst(B);
	VEC ones(B), twoPointNine(B), twoX(B), w0(B), xw0(B),
		epsilon0(B), epsilon1(B), termOne(B), termTwo(B), answer(B);
	funcGetShares(twoPointNine, data_twoPointNine);
	funcGetShares(ones, data_one);

	multiplyByScalar(b, 2, twoX);
	subtractVectors<RSScomputeType>(twoPointNine, twoX, w0, B);
	funcDotProduct(b, w0, xw0, B, true, precision);
	subtractVectors<RSScomputeType>(ones, xw0, epsilon0, B);
	if (PRECISE_DIVISION)
		funcDotProduct(epsilon0, epsilon0, epsilon1, B, true, precision);
	addVectors(ones, epsilon0, termOne, B);
	if (PRECISE_DIVISION)
		addVectors(ones, epsilon1, termTwo, B);
	funcDotProduct(w0, termOne, answer, B, true, precision);
	if (PRECISE_DIVISION)
		funcDotProduct(answer, termTwo, answer, B, true, precision);

	VEC scaledA(batchSize * B), b_repeat(batchSize * B);
	// multiplyByScalar(a, 2, scaledA); //So that a is of precision precision
	for (int i = 0; i < B; ++i)
		for (int j = 0; j < batchSize; ++j)
			b_repeat[i * batchSize + j] = answer[i];
	funcDotProduct(b_repeat, a, quotient, batchSize * B, true, (2 * precision - float_precision)); // Convert to fixed precision
}

template <typename VEC>
void funcMaxpool(VEC &a, VEC &max, RSSVectorSmallType &maxPrime,
				 size_t rows, size_t columns)
{
	log_print("funcMaxpool");
	assert(columns < 256 && "Pooling size has to be smaller than 8-bits");

	size_t size = rows * columns;
	VEC diff(rows);
	RSSVectorSmallType rp(rows), dmpIndexShares(columns * size), temp(size);
	vector<smallType> dmpTemp(columns * size, 0);

	for (int loopCounter = 0; loopCounter < columns; ++loopCounter)
		for (size_t i = 0; i < rows; ++i)
			dmpTemp[loopCounter * rows * columns + i * columns + loopCounter] = 1;
	funcGetShares(dmpIndexShares, dmpTemp);

	for (size_t i = 0; i < size; ++i)
		maxPrime[i] = dmpIndexShares[i];

	for (size_t i = 0; i < rows; ++i)
		max[i] = a[i * columns];

	for (size_t i = 1; i < columns; ++i)
	{
		for (size_t j = 0; j < rows; ++j)
			diff[j] = max[j] - a[j * columns + i];

		funcRELU(diff, rp, max, rows);
		funcSelectBitShares(maxPrime, dmpIndexShares, rp, temp, rows, columns, i);

		for (size_t i = 0; i < size; ++i)
			maxPrime[i] = temp[i];

		for (size_t j = 0; j < rows; ++j)
			max[j] = max[j] + a[j * columns + i];
	}
}

void aggregateCommunication();

// cpp, Malicious
template <typename VEC, typename T>
void funcCheckMaliciousDotProd(const VEC &a, const VEC &b, const VEC &c,
							   const vector<T> &temp, size_t size)
{
	typedef std::pair<T, T> RSSComputeType;
	if (std::is_same<T, myType>::value)
	{
		std::cout << "funcCheckMaliciousDotProd" << std::endl;
	}
	VEC x(size), y(size), z(size);
	PrecomputeObject->getTriplets(x, y, z, size);

	subtractVectors(x, a, x, size);
	subtractVectors(y, b, y, size);

	size_t combined_size = 2 * size;
	VEC combined(combined_size);
	vector<T> rhoSigma(combined_size), rho(size), sigma(size), temp_send(size);
	for (int i = 0; i < size; ++i)
		combined[i] = x[i];

	for (int i = size; i < combined_size; ++i)
		combined[i] = y[i - size];

	funcReconstruct(combined, rhoSigma, combined_size, "rhoSigma", false);

	for (int i = 0; i < size; ++i)
		rho[i] = rhoSigma[i];

	for (int i = size; i < combined_size; ++i)
		sigma[i - size] = rhoSigma[i];

	vector<T> temp_recv(size);
	// Doing x times sigma, rho times y, and rho times sigma
	for (int i = 0; i < size; ++i)
	{
		temp_send[i] = x[i].first + sigma[i];
		temp_send[i] = rho[i] + y[i].first;
		temp_send[i] = rho[i] + sigma[i];
	}

	thread *threads = new thread[2];
	threads[0] = thread(sendVector<T>, ref(temp_send), nextParty(partyNum), size);
	threads[1] = thread(receiveVector<T>, ref(temp_recv), prevParty(partyNum), size);
	for (int i = 0; i < 2; i++)
		threads[i].join();
	delete[] threads;

	for (int i = 0; i < size; ++i)
		if (temp[i] == temp_recv[i])
		{
			// Do the abort thing
		}
}

template <typename Vec, typename T>
void funcCheckMaliciousMatMul(const Vec &a, const Vec &b, const Vec &c,
							  const vector<T> &temp, size_t rows, size_t common_dim, size_t columns,
							  size_t transpose_a, size_t transpose_b)
{
	Vec x(a.size()), y(b.size()), z(c.size());
	PrecomputeObject->getTriplets(x, y, z, rows, common_dim, columns);

	subtractVectors(x, a, x, rows * common_dim);
	subtractVectors(y, b, y, common_dim * columns);

	size_t combined_size = rows * common_dim + common_dim * columns, base_size = rows * common_dim;
	Vec combined(combined_size);
	vector<T> rhoSigma(combined_size), rho(rows * common_dim), sigma(common_dim * columns), temp_send(rows * columns);
	for (int i = 0; i < rows * common_dim; ++i)
		combined[i] = x[i];

	for (int i = rows * common_dim; i < combined_size; ++i)
		combined[i] = y[i - base_size];

	funcReconstruct(combined, rhoSigma, combined_size, "rhoSigma", false);

	for (int i = 0; i < rows * common_dim; ++i)
		rho[i] = rhoSigma[i];

	for (int i = rows * common_dim; i < combined_size; ++i)
		sigma[i - base_size] = rhoSigma[i];

	// Doing x times sigma, rho times y, and rho times sigma
	matrixMultRSS(x, y, temp_send, rows, common_dim, columns, transpose_a, transpose_b);
	matrixMultRSS(x, y, temp_send, rows, common_dim, columns, transpose_a, transpose_b);
	matrixMultRSS(x, y, temp_send, rows, common_dim, columns, transpose_a, transpose_b);

	size_t size = rows * columns;
	vector<T> temp_recv(size);

	thread *threads = new thread[2];
	threads[0] = thread(sendVector<T>, ref(temp_send), nextParty(partyNum), size);
	threads[1] = thread(receiveVector<T>, ref(temp_recv), prevParty(partyNum), size);
	for (int i = 0; i < 2; i++)
		threads[i].join();
	delete[] threads;

	for (int i = 0; i < size; ++i)
		if (temp[i] == temp_recv[i])
		{
			// Do the abort thing
		}
}

// Term by term multiplication of 64-bit vectors overriding precision
template <typename T>
void funcDotProduct(const T &a, const T &b,
					T &c, size_t size, bool truncation, size_t precision)
{
	log_print("funcDotProduct");
	typedef typename std::conditional<std::is_same<T, RSSVectorHighType>::value, highBit, lowBit>::type computeType;
	// if (std::is_same<T, RSSVectorHighType>::value)
	// {
	// 	std::cout << "high" << std::endl;
	// }
	assert(a.size() == size && "Matrix a incorrect for Mat-Mul");
	assert(b.size() == size && "Matrix b incorrect for Mat-Mul");
	assert(c.size() == size && "Matrix c incorrect for Mat-Mul");

	vector<computeType> temp3(size, 0);

	if (truncation == false)
	{
		vector<computeType> recv(size, 0);
		for (int i = 0; i < size; ++i)
		{
			temp3[i] += a[i].first * b[i].first +
						a[i].first * b[i].second +
						a[i].second * b[i].first;
		}

		thread *threads = new thread[2];

		threads[0] = thread(sendVector<computeType>, ref(temp3), prevParty(partyNum), size);
		threads[1] = thread(receiveVector<computeType>, ref(recv), nextParty(partyNum), size);

		for (int i = 0; i < 2; i++)
			threads[i].join();
		delete[] threads;

		for (int i = 0; i < size; ++i)
		{
			c[i].first = temp3[i];
			c[i].second = recv[i];
		}
	}
	else // TODO-trunction
	{
		vector<computeType> diffReconst(size, 0);
		T r(size), rPrime(size);
		PrecomputeObject->getDividedShares(r, rPrime, (1 << precision), size);

		for (int i = 0; i < size; ++i)
		{
			temp3[i] += a[i].first * b[i].first +
						a[i].first * b[i].second +
						a[i].second * b[i].first -
						rPrime[i].first;
		}

		funcReconstruct3out3(temp3, diffReconst, size, "Dot-product diff reconst", false);
		dividePlain(diffReconst, (1l << precision));
		if (partyNum == PARTY_A)
		{
			for (int i = 0; i < size; ++i)
			{
				c[i].first = r[i].first + diffReconst[i];
				c[i].second = r[i].second;
			}
		}

		if (partyNum == PARTY_B)
		{
			for (int i = 0; i < size; ++i)
			{
				c[i].first = r[i].first;
				c[i].second = r[i].second;
			}
		}

		if (partyNum == PARTY_C)
		{
			for (int i = 0; i < size; ++i)
			{
				c[i].first = r[i].first;
				c[i].second = r[i].second + diffReconst[i];
			}
		}
	}
	if (SECURITY_TYPE.compare("Malicious") == 0)
		funcCheckMaliciousDotProd(a, b, c, temp3, size);
}

template <typename Vec, typename T>
void funcDotProduct(const Vec &a, const Vec &b,
					Vec &c, size_t size, bool truncation, size_t precision)
{
	assert(a.size() == size && "Matrix a incorrect for Mat-Mul");
	assert(b.size() == size && "Matrix b incorrect for Mat-Mul");
	assert(c.size() == size && "Matrix c incorrect for Mat-Mul");

	vector<T> temp3(size, 0);

	if (truncation == false)
	{
		vector<T> recv(size, 0);
		for (int i = 0; i < size; ++i)
		{
			temp3[i] += a[i].first * b[i].first +
						a[i].first * b[i].second +
						a[i].second * b[i].first;
		}

		thread *threads = new thread[2];

		threads[0] = thread(sendVector<T>, ref(temp3), prevParty(partyNum), size);
		threads[1] = thread(receiveVector<T>, ref(recv), nextParty(partyNum), size);

		for (int i = 0; i < 2; i++)
			threads[i].join();
		delete[] threads;

		for (int i = 0; i < size; ++i)
		{
			c[i].first = temp3[i];
			c[i].second = recv[i];
		}
	}
	else // TODO-trunction
	{
		vector<T> diffReconst(size, 0);
		Vec r(size), rPrime(size);
		PrecomputeObject->getDividedShares(r, rPrime, (1l << precision), size);

		for (int i = 0; i < size; ++i)
		{
			temp3[i] += a[i].first * b[i].first +
						a[i].first * b[i].second +
						a[i].second * b[i].first -
						rPrime[i].first;
		}

		funcReconstruct3out3(temp3, diffReconst, size, "Dot-product diff reconst", false);
		dividePlain(diffReconst, (1l << precision));
		if (partyNum == PARTY_A)
		{
			for (int i = 0; i < size; ++i)
			{
				c[i].first = r[i].first + diffReconst[i];
				c[i].second = r[i].second;
			}
		}

		if (partyNum == PARTY_B)
		{
			for (int i = 0; i < size; ++i)
			{
				c[i].first = r[i].first;
				c[i].second = r[i].second;
			}
		}

		if (partyNum == PARTY_C)
		{
			for (int i = 0; i < size; ++i)
			{
				c[i].first = r[i].first;
				c[i].second = r[i].second + diffReconst[i];
			}
		}
	}
	if (SECURITY_TYPE.compare("Malicious") == 0)
		funcCheckMaliciousDotProd(a, b, c, temp3, size);
}

template <typename Vec>
void funcMatMul(const Vec &a, const Vec &b, Vec &c,
				size_t rows, size_t common_dim, size_t columns,
				size_t transpose_a, size_t transpose_b, size_t truncation)
{
	log_print("funcMatMul");
	assert(a.size() == rows * common_dim && "Matrix a incorrect for Mat-Mul");
	assert(b.size() == common_dim * columns && "Matrix b incorrect for Mat-Mul");
	assert(c.size() == rows * columns && "Matrix c incorrect for Mat-Mul");

#if (LOG_DEBUG)
	cout << "Rows, Common_dim, Columns: " << rows << "x" << common_dim << "x" << columns << endl;
#endif
	typedef typename std::conditional<std::is_same<Vec, RSSVectorHighType>::value, highBit, lowBit>::type computeType;
	size_t final_size = rows * columns;
	vector<computeType> temp3(final_size, 0), diffReconst(final_size, 0);

	matrixMultRSS(a, b, temp3, rows, common_dim, columns, transpose_a, transpose_b);
	matrixMultRSS(a, b, temp3, rows, common_dim, columns, transpose_a, transpose_b);

	Vec r(final_size), rPrime(final_size);
	PrecomputeObject->getDividedShares(r, rPrime, (1 << truncation), final_size);
	for (int i = 0; i < final_size; ++i)
		temp3[i] = temp3[i] - rPrime[i].first;

	funcReconstruct3out3(temp3, diffReconst, final_size, "Mat-Mul diff reconst", false);
	if (SECURITY_TYPE.compare("Malicious") == 0)
		funcCheckMaliciousMatMul(a, b, c, temp3, rows, common_dim, columns, transpose_a, transpose_b);

	dividePlain(diffReconst, (1l << truncation));

	// for (int i = 0; i < 128; ++i)
	// 	print_linear(diffReconst[i], "FLOAT");
	// cout << endl;

	if (partyNum == PARTY_A)
	{
		for (int i = 0; i < final_size; ++i)
		{
			c[i].first = r[i].first + diffReconst[i];
			c[i].second = r[i].second;
		}
	}

	if (partyNum == PARTY_B)
	{
		for (int i = 0; i < final_size; ++i)
		{
			c[i].first = r[i].first;
			c[i].second = r[i].second;
		}
	}

	if (partyNum == PARTY_C)
	{
		for (int i = 0; i < final_size; ++i)
		{
			c[i].first = r[i].first;
			c[i].second = r[i].second + diffReconst[i];
		}
	}
}

// Debug
void debugMatMul();
void debugDotProd();
void debugPC();
void debugWrap();
void debugReLUPrime();
void debugReLU();
void debugDivision();
void debugBN();
void debugSSBits();
void debugSS();
void debugMaxpool();
/******     Additional Functionalities      *******/
// void debugReduction();
// void debugPartySS();
void debugSquare();
void debugExp();
// void debugSoftmax();

// Test
void testMatMul(size_t rows, size_t common_dim, size_t columns, size_t iter);
void testConvolution(size_t iw, size_t ih, size_t Din, size_t Dout,
					 size_t f, size_t S, size_t P, size_t B, size_t iter);
void testRelu(size_t r, size_t c, size_t iter);
void testReluPrime(size_t r, size_t c, size_t iter);
void testMaxpool(size_t ih, size_t iw, size_t Din, size_t f, size_t S, size_t B, size_t iter);
void testReduction(size_t size);

// #include "Functionalities_impl.h"
// template void funcReconstruct<RSSVectorLowType, lowBit>(const RSSVectorLowType &a, vector<lowBit> &b, size_t size, string str, bool print);
// template void funcReconstruct<RSSVectorHighType, highBit>(const RSSVectorHighType &a, vector<highBit> &b, size_t size, string str, bool print);
// template void funcReconstruct<RSSVectorSmallType, smallType>(const RSSVectorSmallType &a, vector<smallType> &b, size_t size, string str, bool print);
// Move here from tools.cpp
// template <typename Vec>
// void print_vector(Vec &var, string type, string pre_text, int print_nos);
// template <typename Vec>
// void print_vector(Vec &var, string type, string pre_text, int print_nos)
// {
// 	if (std::is_same<Vec, RSSVectorSmallType>::value)
// 	{
// 		size_t temp_size = var.size();
// 		vector<smallType> b(temp_size);
// 		funcReconstruct(var, b, temp_size, "anything", false);
// 		cout << pre_text << endl;
// 		for (int i = 0; i < print_nos; ++i)
// 		{
// 			cout << b[i] << " ";
// 			// if (i % 10 == 9)
// 			// std::cout << std::endl;
// 		}
// 		cout << endl;
// 	}
// 	else
// 	{
// 		size_t temp_size = var.size();
// 		typedef lowBit T;
// 		if (std::is_same<Vec, RSSVectorHighType>::value)
// 			typedef highBit T;
// 		vector<T> b(temp_size);
// 		funcReconstruct(var, b, temp_size, "anything", false);
// 		cout << pre_text << endl;
// 		for (int i = 0; i < print_nos; ++i)
// 		{
// 			print_linear(b[i], type);
// 			// if (i % 10 == 9)
// 			// std::cout << std::endl;
// 		}
// 		cout << endl;
// 	}
// }
void print_vector(RSSVectorLowType &var, string type, string pre_text, int print_nos);
void print_vector(RSSVectorHighType &var, string type, string pre_text, int print_nos);
void print_vector(RSSVectorSmallType &var, string type, string pre_text, int print_nos);

template <typename Vec, typename T>
void funcAddOneConst(Vec &result, T c, size_t size)
{
	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; i++)
		{
			result[i].first += c;
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (size_t i = 0; i < size; i++)
		{
			result[i].second += c;
		}
	}
}

template <typename Vec, typename T>
void funcAddOneConst(Vec &result, const Vec &input, T c, size_t size)
{
	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; i++)
		{
			result[i] = make_pair(input[i].first + c, input[i].second);
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (size_t i = 0; i < size; i++)
		{
			result[i] = make_pair(input[i].first, input[i].second + c);
		}
	}
	else
	{
		for (size_t i = 0; i < size; i++)
		{
			result[i] = make_pair(input[i].first, input[i].second);
		}
	}
}

template <typename vec, typename T>
void funcAddConst(vec &result, vector<T> c, size_t size)
{
	if (partyNum == PARTY_B)
	{
		for (size_t i = 0; i < size; i++)
		{
			result[i].second += c[i];
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (size_t i = 0; i < size; i++)
		{
			result[i].fisrt += c[i];
		}
	}
}

template <typename vec>
void funcAdd(vec &result, vec &data1, vec &data2, size_t size, bool minus)
{
	if (minus)
	{
		for (size_t i = 0; i < size; i++)
		{
			result[i] = make_pair(data1[i].first - data2[i].first, data1[i].second - data2[i].second);
		}
	}
	else
	{
		for (size_t i = 0; i < size; i++)
		{
			result[i] = make_pair(data1[i].first + data2[i].first, data1[i].second + data2[i].second);
		}
	}
}

/********************* Share Conversion Functionalites *********************/
template <typename Vec, typename T>
void funcRandBitByXor(Vec &b, size_t size);

void funcReduction(RSSVectorLowType &output, const RSSVectorHighType &input, size_t size);
void funcWCExtension(RSSVectorHighType &output, const RSSVectorLowType &input, size_t size);
void funcMSExtension(RSSVectorHighType &output, const RSSVectorLowType &input, size_t size);
void funcPosWrap(RSSVectorHighType &w, const RSSVectorLowType &input, size_t size);
void funcMixedShareGen(RSSVectorHighType &an, RSSVectorLowType &am, RSSVectorHighType &msb, size_t size);
template <typename Vec, typename T>
void funcProbTruncation(Vec &output, const Vec &input, int trunc_bits, size_t size);

/**
 * @brief trunc trunc_bits of input to output
 *
 * @tparam Vec
 * @tparam T
 * @param output
 * @param input
 * @param trunc_bits
 * @param size
 */
template <typename Vec, typename T>
void funcProbTruncation(Vec &output, const Vec &input, int trunc_bits, size_t size)
{
	size_t k = sizeof(T) << 3;
	assert(k - 2 > trunc_bits);
	size_t reall = k - trunc_bits;
	T bias1 = (1l << (k - 2));
	T bias2 = -(1l << (k - 2 - trunc_bits));
	T msb = (1l << (k - 1));

	Vec rbits(size * k);
	Vec r(size);
	Vec rtrunc(size);
	Vec rmsb(size);
	vector<T> z(size);

	if (OFFLINE_ON) // get r and rtrunc
	{
		funcRandBitByXor<Vec, T>(rbits, rbits.size());
		T temp1;
		T temp2;
		for (size_t i = 0; i < size; ++i)
		{
			rmsb[i] = rbits[i * k];
			temp1 = 0;
			temp2 = 0;
			// msb1 = rmsb[i].first << reall;
			// msb2 = rmsb[i].second << reall;
			size_t j;
			for (j = 0; j < reall; ++j)
			{
				temp1 = (temp1 << 1) + rbits[i * k + j].first;
				temp2 = (temp2 << 1) + rbits[i * k + j].second;
			}
			rtrunc[i] = make_pair(temp1, temp2);
			for (; j < k; ++j)
			{
				temp1 = (temp1 << 1) + rbits[i * k + j].first;
				temp2 = (temp2 << 1) + rbits[i * k + j].second;
			}
			r[i] = make_pair(temp1, temp2);
		}
	}

	// log
	// vector<T> r_p(size);
	// vector<T> rtrunc_p(size);
	// vector<T> rmsb_p(size);
	// vector<T> rbits_p(rbits.size());
	// funcReconstruct<Vec, T>(r, r_p, size, "r", false);
	// funcReconstruct<Vec, T>(rtrunc, rtrunc_p, size, "rtrunc", false);
	// funcReconstruct<Vec, T>(rmsb, rmsb_p, size, "rmsb", false);
	// printVector<T>(r_p, "r", size);
	// printHighBitVec(r_p, "r", size);
	// printVector<T>(r_p, "r", size);
	// printHighBitVec(rtrunc_p, "rtrunc", size);
	// printVector<T>(rtrunc_p, "rtrunc", size);
	// printVector<T>(rtrunc_p, "rtrunc", size);
	//  printVector<T>(rmsb_p, "rmsb", size);
	Vec input2(size);
	funcAddOneConst<Vec, T>(input2, input, bias1, size);
	// funcAddOneConst(input, bias1, size);

	// log x
	// vector<T> biasx(size);
	// funcReconstruct(input, biasx, size, "bias x", false);
	// printVector<T>(biasx, "x", size);
	// printHighBitVec(biasx, "x", size);

	// reveal x+r
	funcAdd(input2, input2, r, size, false);
	funcReconstruct(input2, z, size, "x+r", false);

	// log x+r
	// printVector<T>(z, "x+r", size);
	// printHighBitVec(z, "x+r", size);

	for (size_t i = 0; i < size; ++i)
	{
		// wrap,  [w]k = [rk−1]k · ¬⌊z/2k−1⌋
		if ((z[i] & msb))
		{
			rmsb[i] = make_pair(0, 0); // w-->rmsb
		}
		else
		{
			rmsb[i] = make_pair((rmsb[i].first << reall), (rmsb[i].second << reall));
		}
	}

	// 	log func
	// vector<T> w(size);
	// funcReconstruct<Vec, T>(rmsb, w, size, "w", true);

	//  [x] = z − [rtrunc] + [w]·2^(k−f)
	// if (partyNum == PARTY_A)
	// {
	// 	for (size_t i = 0; i < size; i++)
	// 	{
	// 		output[i] = make_pair((z[i] >> trunc_bits), 0);
	// 	}
	// }
	// else if (partyNum == PARTY_C)
	// {
	// 	for (size_t i = 0; i < size; i++)
	// 	{
	// 		output[i] = make_pair(0, (z[i] >> trunc_bits));
	// 	}
	// }
	// else
	// {
	// 	for (size_t i = 0; i < size; i++)
	// 	{
	// 		output[i] = make_pair(0, 0);
	// 	}
	// }

	// log
	// vector<T> xtrunc(size);
	// funcReconstruct<Vec, T>(output, xtrunc, size, "ztrunc", false);
	// // printHighBitVec(xtrunc, "", size);
	// printVector(xtrunc, "ztrunc", size);

	// funcAdd(output, output, rtrunc, size, true); // truncz-truncr

	// log
	// funcReconstruct<Vec, T>(output, xtrunc, size, "ztrunc-rtrunc", true);
	// // printHighBitVec(xtrunc, "", size);
	// printVector(xtrunc, "z-r", size);

	// funcAdd(output, output, rmsb, size, false); // x' += wrap

	// log
	// vector<T> x(size);
	// funcReconstruct<Vec, T>(output, x, size, "before bias", false);
	// printVector(x, "before bias", size);
	// printHighBitVec(x, "before bias", size);
	// cout << bitset<64>(bias2) << endl;

	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; i++)
		{
			output[i] = make_pair((z[i] >> trunc_bits) + rmsb[i].first - rtrunc[i].first, rmsb[i].second - rtrunc[i].second);
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (size_t i = 0; i < size; i++)
		{
			output[i] = make_pair(rmsb[i].first - rtrunc[i].first, (z[i] >> trunc_bits) + rmsb[i].second - rtrunc[i].second);
		}
	}
	else
	{
		for (size_t i = 0; i < size; i++)
		{
			output[i] = make_pair(rmsb[i].first - rtrunc[i].first, rmsb[i].second - rtrunc[i].second);
		}
	}

	funcAddOneConst(output, bias2, size);

	// log
	// funcReconstruct<Vec, T>(output, x, size, "after bias", true);
	// printVector(x, "after bias", size);
	// printHighBitVec(x, "after bias", size);
}

template <typename Vec, typename T>
void funcProbTruncation(Vec &data, int trunc_bits, size_t size)
{
	size_t k = sizeof(T) << 3;
	assert(k - 2 > trunc_bits);
	size_t reall = k - trunc_bits;
	T bias1 = (1l << (k - 2));
	T bias2 = -(1l << (k - 2 - trunc_bits));
	T msb = (1l << (k - 1));

	Vec rbits(size * k);
	Vec r(size);
	Vec rtrunc(size);
	Vec rmsb(size);
	vector<T> z(size);

	if (OFFLINE_ON) // get r and rtrunc
	{
		funcRandBitByXor<Vec, T>(rbits, rbits.size());
		T temp1;
		T temp2;
		for (size_t i = 0; i < size; ++i)
		{
			rmsb[i] = rbits[i * k];
			temp1 = 0;
			temp2 = 0;
			// msb1 = rmsb[i].first << reall;
			// msb2 = rmsb[i].second << reall;
			size_t j;
			for (j = 0; j < reall; ++j)
			{
				temp1 = (temp1 << 1) + rbits[i * k + j].first;
				temp2 = (temp2 << 1) + rbits[i * k + j].second;
			}
			rtrunc[i] = make_pair(temp1, temp2);
			for (; j < k; ++j)
			{
				temp1 = (temp1 << 1) + rbits[i * k + j].first;
				temp2 = (temp2 << 1) + rbits[i * k + j].second;
			}
			r[i] = make_pair(temp1, temp2);
		}
	}

	// log
	// vector<T> r_p(size);
	// vector<T> rtrunc_p(size);
	// vector<T> rmsb_p(size);
	// vector<T> rbits_p(rbits.size());
	// funcReconstruct<Vec, T>(r, r_p, size, "r", false);
	// funcReconstruct<Vec, T>(rtrunc, rtrunc_p, size, "rtrunc", false);
	// funcReconstruct<Vec, T>(rmsb, rmsb_p, size, "rmsb", false);
	// printVector<T>(r_p, "r", size);
	// printHighBitVec(r_p, "r", size);
	// printVector<T>(r_p, "r", size);
	// printHighBitVec(rtrunc_p, "rtrunc", size);
	// printVector<T>(rtrunc_p, "rtrunc", size);
	// printVector<T>(rtrunc_p, "rtrunc", size);
	//  printVector<T>(rmsb_p, "rmsb", size);
	funcAddOneConst<Vec, T>(data, bias1, size);
	// funcAddOneConst(input, bias1, size);

	// log x
	// vector<T> biasx(size);
	// funcReconstruct(input, biasx, size, "bias x", false);
	// printVector<T>(biasx, "x", size);
	// printHighBitVec(biasx, "x", size);

	// reveal x+r
	funcAdd(data, data, r, size, false);
	funcReconstruct(data, z, size, "x+r", false);

	// log x+r
	// printVector<T>(z, "x+r", size);
	// printHighBitVec(z, "x+r", size);

	for (size_t i = 0; i < size; ++i)
	{
		// wrap,  [w]k = [rk−1]k · ¬⌊z/2k−1⌋
		if ((z[i] & msb))
		{
			rmsb[i] = make_pair(0, 0); // w-->rmsb
		}
		else
		{
			rmsb[i] = make_pair((rmsb[i].first << reall), (rmsb[i].second << reall));
		}
	}

	// 	log func
	// vector<T> w(size);
	// funcReconstruct<Vec, T>(rmsb, w, size, "w", true);

	//  [x] = z − [rtrunc] + [w]·2^(k−f)
	// if (partyNum == PARTY_A)
	// {
	// 	for (size_t i = 0; i < size; i++)
	// 	{
	// 		output[i] = make_pair((z[i] >> trunc_bits), 0);
	// 	}
	// }
	// else if (partyNum == PARTY_C)
	// {
	// 	for (size_t i = 0; i < size; i++)
	// 	{
	// 		output[i] = make_pair(0, (z[i] >> trunc_bits));
	// 	}
	// }
	// else
	// {
	// 	for (size_t i = 0; i < size; i++)
	// 	{
	// 		output[i] = make_pair(0, 0);
	// 	}
	// }

	// log
	// vector<T> xtrunc(size);
	// funcReconstruct<Vec, T>(output, xtrunc, size, "ztrunc", false);
	// // printHighBitVec(xtrunc, "", size);
	// printVector(xtrunc, "ztrunc", size);

	// funcAdd(output, output, rtrunc, size, true); // truncz-truncr

	// log
	// funcReconstruct<Vec, T>(output, xtrunc, size, "ztrunc-rtrunc", true);
	// // printHighBitVec(xtrunc, "", size);
	// printVector(xtrunc, "z-r", size);

	// funcAdd(output, output, rmsb, size, false); // x' += wrap

	// log
	// vector<T> x(size);
	// funcReconstruct<Vec, T>(output, x, size, "before bias", false);
	// printVector(x, "before bias", size);
	// printHighBitVec(x, "before bias", size);
	// cout << bitset<64>(bias2) << endl;

	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; i++)
		{
			data[i] = make_pair((z[i] >> trunc_bits) + rmsb[i].first - rtrunc[i].first, rmsb[i].second - rtrunc[i].second);
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (size_t i = 0; i < size; i++)
		{
			data[i] = make_pair(rmsb[i].first - rtrunc[i].first, (z[i] >> trunc_bits) + rmsb[i].second - rtrunc[i].second);
		}
	}
	else
	{
		for (size_t i = 0; i < size; i++)
		{
			data[i] = make_pair(rmsb[i].first - rtrunc[i].first, rmsb[i].second - rtrunc[i].second);
		}
	}

	funcAddOneConst(data, bias2, size);

	// log
	// funcReconstruct<Vec, T>(output, x, size, "after bias", true);
	// printVector(x, "after bias", size);
	// printHighBitVec(x, "after bias", size);
}

// void funcProbTruncation(Vec &output, Vec &input, int trunc_bits, size_t size)
void funcTruncAndReduce(RSSVectorLowType &a, const RSSVectorHighType &b, int trunc_bits, size_t size);

/********************* Mixed-Precision Activations Functionalites *********************/
void funcMReLU();

/**
 * Functionality for Softmax computation
 * */

// Compute b*b
template <typename Vec>
void funcSquare(const Vec &a, Vec &b, size_t size)
{
	log_print("funcSquare");

	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<Vec, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<Vec, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(a).name() << endl;
	}

	typedef typename std::conditional<std::is_same<Vec, RSSVectorHighType>::value, highBit, lowBit>::type elementType;
	vector<elementType> temp3(size, 0), diffReconst(size, 0);

	for (size_t i = 0; i < size; i++)
	{
		temp3[i] += a[i].first * a[i].first +
					a[i].first * a[i].second +
					a[i].second * a[i].first;
	}

	// TODO: perform truncation
	Vec r(size), rPrime(size);
	PrecomputeObject->getDividedShares(r, rPrime, (1 << float_precision), size);

	for (size_t i = 0; i < size; i++)
	{
		temp3[i] -= rPrime[i].first;
	}

	funcReconstruct3out3(temp3, diffReconst, size, "Square Truncation", false);
	dividePlain(diffReconst, (1l << float_precision));

	// cout << "Reconstrut Square Diff." << endl;
	// for (size_t i = 0; i < size; i++) {
	// 	print_linear(diffReconst[i], "FLOAT");
	// }
	// cout << endl;

	// Reshare 2-3 RSS
	if (partyNum == PARTY_A)
	{
		for (int i = 0; i < size; ++i)
		{
			b[i].first = r[i].first + diffReconst[i];
			b[i].second = r[i].second;
		}
	}

	if (partyNum == PARTY_B)
	{
		for (int i = 0; i < size; ++i)
		{
			b[i].first = r[i].first;
			b[i].second = r[i].second;
		}
	}

	if (partyNum == PARTY_C)
	{
		for (int i = 0; i < size; ++i)
		{
			b[i].first = r[i].first;
			b[i].second = r[i].second + diffReconst[i];
		}
	}
}

// Reference from CryptGPU:https://eprint.iacr.org/2021/533.pdf
template <typename Vec>
void funcExp(const Vec &a, Vec &b, size_t size)
{
	log_print("funcExp");

	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<Vec, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<Vec, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(a).name() << endl;
	}

	typedef typename std::conditional<std::is_same<Vec, RSSVectorHighType>::value, highBit, lowBit>::type elementType;
	vector<elementType> diffReconst(size, 0);

	// compute FXP(x/m)
	// vector<elementType> temp3(size, 0), 
	// for (size_t i = 0; i < size; i++)
	// {
	// 	temp3[i] = a[i].first;
	// }

	// TODO: perform truncation
	Vec r(size), rPrime(size);
	PrecomputeObject->getDividedShares(r, rPrime, (1 << EXP_PRECISION), size);

	Vec temp = a;
	for (size_t i = 0; i < size; i++)
	{
		// temp3[i] -= rPrime[i].first;
		temp[i] =  temp[i] - rPrime[i];
	}
	funcReconstruct(temp, diffReconst, size, "Exp Truncation", false);

	// funcReconstruct3out3(temp3, diffReconst, size, "Exp Truncation", false);
	dividePlain(diffReconst, (1l << EXP_PRECISION));

	// cout << "Reconstrut Exp Diff." << endl;
	// for (size_t i = 0; i < size; i++) {
	// 	print_linear(diffReconst[i], "FLOAT");
	// }
	// cout << endl;

	// !!!!!!!!!!!!!!! Note: the below computes r + (x'-r')/2^d + 1. The last +1 operation is required to compute 1+x/m !!!!!!!!! 
	if (partyNum == PARTY_A)
	{
		for (int i = 0; i < size; ++i)
		{
			b[i].first = r[i].first + diffReconst[i] + (1 << float_precision);
			b[i].second = r[i].second;
		}
	}

	if (partyNum == PARTY_B)
	{
		for (int i = 0; i < size; ++i)
		{
			b[i].first = r[i].first;
			b[i].second = r[i].second;
		}
	}

	if (partyNum == PARTY_C)
	{
		for (int i = 0; i < size; ++i)
		{
			b[i].first = r[i].first;
			b[i].second = r[i].second + diffReconst[i] + (1 << float_precision);
		}
	}

	// compute (1+x/m)^m. using m invocations of square
	for (size_t i = 0; i < EXP_PRECISION; i++)
	{
		funcSquare(b, b, size);
	}
}

/**
 * @brief get the reciprocal of b
 *  'NR' : Newton-Raphson method computes the reciprocal using iterations of :math:
 *  x_{i+1} = (2x_i - self * x_i^2) and uses math:
 *  3*exp(1 - 2x) + 0.003` as an initial guess by default
 *
 * @tparam VEC
 * @param a result
 * @param b input
 * @param input_in_01 Allows a user to indicate that the input is in the range [0, 1],
					causing the function optimize for this range. This is useful for improving
					the accuracy of functions on probabilities (e.g. entropy functions).
 * @param size
 */
template <typename VEC>
void funcReciprocal(VEC &a, const VEC &b, bool input_in_01,
					size_t size)
{
	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<VEC, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<VEC, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(a).name() << endl;
	}
	const highBit rec_const = 0.003 * (1 << float_precision);

	VEC temp(size);
	if (input_in_01)
	{
		// funcMulConst(b, b, 64, size);
		// funcReciprocal(a, b, false, size);
		// funcMulConst(a, a, 64, size);
		return;
	}

	// result = 3 * (1 - 2 * b).exp() + 0.003
	// b = (1 - 2 * b)
	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; i++)
		{
			a[i].first = FLOAT_BIAS - 2 * b[i].first;
			a[i].second = -2 * b[i].second;
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (size_t i = 0; i < size; i++)
		{
			a[i].first = -2 * b[i].first;
			a[i].second = FLOAT_BIAS - 2 * b[i].second;
		}
	}
	else
	{
		for (size_t i = 0; i < size; i++)
		{
			a[i].first = -2 * b[i].first;
			a[i].second = -2 * b[i].second;
		}
	}

	funcExp(a, a, size); // a = exp(a)= exp(1 - 2 * b)

	// a = 3 * (1 - 2 * b).exp() + 0.003
	// 0.003 * (1<<13) = 24.576
	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; i++)
		{
			a[i].first = 3 * a[i].first + rec_const;
			a[i].second = 3 * a[i].second;
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (size_t i = 0; i < size; i++)
		{
			a[i].first = 3 * a[i].first;
			a[i].second = 3 * a[i].second + rec_const;
		}
	}
	else
	{
		for (size_t i = 0; i < size; i++)
		{
			a[i].first = 3 * a[i].first;
			a[i].second = 3 * a[i].second;
		}
	}

	// funcReconstruct(a, r, size, "3*x + 0.003", false);
	// printVectorReal(r, "3*x + 0.003", size);

	// x_{i+1} = (2x_i - self * x_i^2)
	for (size_t j = 0; j < REC_ITERS; ++j)
	{
		funcSquare(a, temp, size);									// temp = a*a
		funcDotProduct(temp, b, temp, size, true, float_precision); // temp = a*a*b
		for (size_t i = 0; i < size; ++i)
		{
			a[i].first = (a[i].first << 1) - temp[i].first;
			a[i].second = (a[i].second << 1) - temp[i].second;
			// a[i].first = 2 * a[i].first - a[i].first * a[i].first * b[i].first;
			// a[i].second = 2 * a[i].second - a[i].second * a[i].second * b[i].second;
		}
		// funcReconstruct(a, r, size, "it", false);
		// printVectorReal(r, "it", size);
	}
}

template <typename VEC>
void funcReciprocal2(VEC &a, const VEC &b, bool input_in_01,
					 size_t size)
{
	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<VEC, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<VEC, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(a).name() << endl;
	}
	VEC temp(size);
	if (input_in_01)
	{
		// funcMulConst(b, b, 64, size);
		// funcReciprocal(a, b, false, size);
		// funcMulConst(a, a, 64, size);
		return;
	}

	// result = 3 * (1 - 2 * b).exp() + 0.003
	// b = (1 - 2 * b)
	// if (OFFLINE_ON)
	// {
		if (partyNum == PARTY_A)
		{
			for (size_t i = 0; i < size; i++)
			{
				a[i].first = (1 << (float_precision - REC_Y));
				a[i].second = 0;
			}
		}
		else if (partyNum == PARTY_C)
		{
			for (size_t i = 0; i < size; i++)
			{
				a[i].first = 0;
				a[i].second = (1 << (float_precision - REC_Y));
			}
		}
		else
		{
			for (size_t i = 0; i < size; i++)
			{
				a[i].first = 0;
				a[i].second = 0;
			}
		}
	// }

	// a = 3 * (1 - 2 * b).exp() + 0.003
	// 0.003 * (1<<13) = 24.576

	// funcReconstruct(a, r, size, "3*x + 0.003", false);
	// printVectorReal(r, "3*x + 0.003", size);

	// x_{i+1} = (2x_i - self * x_i^2)
	for (size_t j = 0; j < REC_Y; ++j)
	{
		funcSquare(a, temp, size);									// temp = a*a
		funcDotProduct(temp, b, temp, size, true, FLOAT_PRECISION); // temp = a*a*b
		for (size_t i = 0; i < size; ++i)
		{
			a[i].first = (a[i].first << 1) - temp[i].first;
			a[i].second = (a[i].second << 1) - temp[i].second;
			// a[i].first = 2 * a[i].first - a[i].first * a[i].first * b[i].first;
			// a[i].second = 2 * a[i].second - a[i].second * a[i].second * b[i].second;
		}
		// funcReconstruct(a, r, size, "it", false);
		// printVectorReal(r, "it", size);
	}
}

template <typename VEC>
void funcDivisionByNR(VEC &result, const VEC &input, const VEC &quotient,
					  size_t size)
{

	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<VEC, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<VEC, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(input).name() << endl;
	}
	VEC q_rec(size);

	if constexpr (std::is_same<VEC, RSSVectorLowType>::value && MP_FOR_DIVISION) {
		cout << "Mixed-Precision Division" << endl;
		RSSVectorHighType highP_dividend(size), highP_rec(size);
		funcMSExtension(highP_dividend, quotient, size);
		funcMulConst(highP_dividend, highP_dividend, 1 << (HIGH_PRECISION - LOW_PRECISION), size);	// maintain same precision
		funcReciprocal2(highP_rec, highP_dividend, false, size);
		funcTruncAndReduce(q_rec, highP_rec, (HIGH_PRECISION - LOW_PRECISION), size);
	} else {
		funcReciprocal2(q_rec, quotient, false, size);
	}

	funcDotProduct(q_rec, input, result, size, true, float_precision);
}

/**
 * @brief   Computes the inverse square root of the input using the Newton-Raphson method.
 *
 * @tparam Vec
 * @param result
 * @param input
 * @param size
 */
template <typename Vec, typename T>
void funcInverseSqrt(Vec &result, const Vec &input, size_t size)
{

	// Initialize using decent approximation
	// y = exp(-( x/2 + 0.2 )) * 2.2 + 0.2
	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<Vec, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<Vec, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(input).name() << endl;
	}
	const T insqrt_a0 = 0.2 * (1 << float_precision);
	const T insqrt_a3 = 3 * (1 << float_precision);
	Vec temp(size);
	funcProbTruncation<Vec, T>(result, input, 1, size); // x/2
	if (partyNum == PARTY_A)
	{ // -( x/2 + 0.2 )
		for (int i = 0; i < size; ++i)
		{
			result[i].first = -result[i].first - insqrt_a0;
			result[i].second = -result[i].second;
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (int i = 0; i < size; ++i)
		{
			result[i].first = -result[i].first;
			result[i].second = -result[i].second - insqrt_a0;
		}
	}
	else
	{
		for (int i = 0; i < size; ++i)
		{
			result[i].first = -result[i].first;
			result[i].second = -result[i].second;
		}
	}
	// funcReconstruct(result, plain, size, "-(x/2 + 0.2)", true);
	// printVectorReal(plain, "-(x/2 + 0.2)", size);

	funcExp(result, result, size); // exp(-( x/2 + 0.2 ))

	// exp(-( x/2 + 0.2 )) * 2 + 0.2
	// y -= x/1024
	funcProbTruncation<Vec, T>(temp, input, 10, size);

	if (partyNum == PARTY_A)
	{
		for (int i = 0; i < size; ++i)
		{
			result[i].first = (result[i].first << 1) + insqrt_a0 - temp[i].first;
			result[i].second = (result[i].second << 1) - temp[i].second;
		}
	}
	else if (partyNum == PARTY_C)
	{
		for (int i = 0; i < size; ++i)
		{
			result[i].first = (result[i].first << 1) - temp[i].first;
			result[i].second = (result[i].second << 1) + insqrt_a0 - temp[i].second;
		}
	}
	else
	{
		for (int i = 0; i < size; ++i)
		{
			result[i].first = (result[i].first << 1) - temp[i].first;
			result[i].second = (result[i].second << 1) - temp[i].second;
		}
	}

	// funcAdd<Vec>(result, result, temp, size, true);

	// Newton Raphson iterations for inverse square root
	for (size_t i = 0; i < INVSQRT_ITERS; i++)
	{																	// y = (y * (3 - x * y * y))/2
		funcSquare(result, temp, size);									// y1 = y*y
		funcDotProduct(temp, input, temp, size, true, float_precision); // y2 = y*y*x

		if (partyNum == PARTY_A)
		{ // 3 - x * y * y
			for (int i = 0; i < size; ++i)
			{
				temp[i].first = -temp[i].first + insqrt_a3;
				temp[i].second = -temp[i].second;
			}
		}
		else if (partyNum == PARTY_C)
		{
			for (int i = 0; i < size; ++i)
			{
				temp[i].first = -temp[i].first;
				temp[i].second = -temp[i].second + insqrt_a3;
			}
		}
		else
		{
			for (int i = 0; i < size; ++i)
			{
				temp[i].first = -temp[i].first;
				temp[i].second = -temp[i].second;
			}
		}

		// result = (y * (3 - x * y * y))/2
		funcDotProduct(temp, result, result, size, true, float_precision + 1);
		// funcProbTruncation<Vec, T>(result, result, 1, size);
	}
}

template<typename Vec, typename T>
void mixedPrecisionOp(Vec &output, const Vec &input, size_t size) {
	// inver Square Root
	// https://stackoverflow.com/questions/63469333/why-does-the-false-branch-of-if-constexpr-get-compiled
	if constexpr (MP_FOR_INV_SQRT && std::is_same<Vec, RSSVectorLowType>::value) {
		cout << "Mixed-Precision Inverse Sqrt" << endl;
		RSSVectorHighType highP_var_eps(size), highP_inv_sqrt(size);
		funcMSExtension(highP_var_eps, input, size);
		funcMulConst(highP_var_eps, highP_var_eps, 1 << (HIGH_PRECISION - LOW_PRECISION), size);	// maintain same precision
		funcInverseSqrt<RSSVectorHighType, highBit>(highP_inv_sqrt, highP_var_eps, size); // [1,D]
		funcTruncAndReduce(output, highP_inv_sqrt, (HIGH_PRECISION - LOW_PRECISION), size);
	} else {
		funcInverseSqrt<Vec, T>(output, input, size); // [1,D]
	}
}

// Reference from CryptGPU:https://eprint.iacr.org/2021/533.pdf
template <typename Vec>
void funcSoftmax(const Vec &a, Vec &b, size_t rows, size_t cols, bool masked)
{
	log_print("funcSoftmax");
	typedef typename std::conditional<std::is_same<Vec, RSSVectorHighType>::value, highBit, lowBit>::type elementType;

	size_t size = rows * cols;

	// normalize the input
	Vec temp(size), max(rows);
	RSSVectorSmallType maxPrime(size);

	temp = a;
	funcMaxpool(temp, max, maxPrime, rows, cols);

	// vector<elementType> temp_reconst(rows);
	// funcReconstruct(max, temp_reconst, rows, "Softmax Log", true);

	// vector<smallType> reconst_maxPrime(maxPrime.size());
	// funcReconstructBit(maxPrime, reconst_maxPrime, rows * cols, "maxP", true);

	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			temp[i * cols + j].first -= max[i].first;
			temp[i * cols + j].second -= max[i].second;
		}
	}

	// compute exp of each element
	Vec exp_elements(size);
	funcExp(temp, exp_elements, size);

	// vector<elementType> reconst_exp_elements(size);
	// funcReconstruct(exp_elements, reconst_exp_elements, size, "exp", true);

	// compute the dividend, i.e., the sum of the exps
	Vec dividend(size);
	for (size_t i = 0; i < rows; i++)
	{
		elementType tmp_sum_first = 0, tmp_sum_second = 0;
		for (size_t j = 0; j < cols; j++)
		{
			tmp_sum_first += exp_elements[i * cols + j].first;
			tmp_sum_second += exp_elements[i * cols + j].second;
		}
		for (size_t j = 0; j < cols; j++)
		{
			dividend[i * cols + j].first = tmp_sum_first;
			dividend[i * cols + j].second = tmp_sum_second;
		}
	}

	// vector<elementType> reconst_dividend(size);
	// funcReconstruct(dividend, reconst_dividend, size, "dividend", true);

	// compute the division
	funcDivisionByNR(b, exp_elements, dividend, size);
}

// Random shared bit [b] over ring of Vec, and b is 0/1;
// in fact, this is supposed to be offline op
template <typename Vec, typename T, typename RealVec>
void funcRandBit(RealVec &b, size_t size)
{
	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<Vec, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<Vec, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(b).name() << endl;
	}
	Vec a(size);
	Vec btemp(size);
	// Vec btemp(size);
	vector<T> e(size);
	int K = sizeof(T) << 3;
	int k1 = K - 1;
	bool isFailure = false;

	// test data
	// vector<T> a_p(size);

	do
	{
		PrecomputeObject->getPairRand<Vec, T>(a, size);

		for (size_t i = 0; i < size; i++) // a = 2u + 1
		{
			a[i].first = (a[i].first << 1) + 1;
			a[i].second = (a[i].second << 1) + 1;
		}
		// test log
		// funcReconstruct<Vec, T>(a, a_p, size, "a", false);

		funcDotProduct<Vec, T>(a, a, btemp, size, false, float_precision); // e = a*a
		funcReconstruct<Vec, T>(btemp, e, size, "a*a", false);			   // reveal e

		// test log
		// printVector<T>(a_p, "a", size);
		// printVector<T>(e, "e=a*a", size);

		for (size_t i = 0; i < size; i++)
		{
			if ((e[i] & 1) == 0) // a is not odd
			{
				isFailure = true;
				cout << "failure" << endl;
				break;
			}
		}

	} while (isFailure);

	T temp;
	for (int i = 0; i < size; i++)
	{
		temp = sqrRoot<T>(e[i], K); // c=e^(1/2)
		e[i] = invert<T>(temp, k1); // c^(-1)
	}

	// e [d] ← c−1[a] + 1.
	switch (partyNum)
	{
	case PARTY_A:
		for (size_t i = 0; i < size; i++)
		{
			btemp[i] = make_pair((a[i].first * e[i] + 1), (a[i].second * e[i] + 1));
		}
		break;
	case PARTY_C:
		for (size_t i = 0; i < size; i++)
		{
			btemp[i] = make_pair((a[i].first * e[i] - 1), (a[i].second * e[i] + 1));
		}
		break;
	default:
		for (size_t i = 0; i < size; i++)
		{
			btemp[i] = make_pair((a[i].first * e[i] + 1), (a[i].second * e[i] - 1));
		}
		break;
	}

	// logf
	// vector<longBit> ac1(size);
	// funcReconstruct<RSSVectorLongType, longBit>(btemp, ac1, size, "a*c+1", false);
	// printVector<longBit>(ac1, "a*c+1", size);

	for (size_t i = 0; i < size; i++)
	{
		b[i] = make_pair(btemp[i].first >> 1, btemp[i].second >> 1);
	}
	// funcReconstruct<RSSVectorLongType, longBit>(btemp, ac1, size, "a*c+1", false);
	// printVector<longBit>(ac1, "after 1/2", size);
}

/******************** Boolean share op ******************/
void mergeBoolVec(RSSVectorBoolType &result, const vector<bool> &a1, const vector<bool> &a2, size_t size);
void mergeRSSVectorBool(vector<bool> &result, RSSVectorBoolType &data, size_t size);
void funcBoolShareSender(RSSVectorBoolType &result, const vector<bool> &data, size_t size);
void funcBoolShareReceiver(RSSVectorBoolType &result, int shareParty, size_t size);

// void funcBoolShare(RSSVectorBoolType &result, const vector<bool> &a, size_t size);

void mergeBoolVec(RSSVectorBoolType &result, const vector<bool> &a1, const vector<bool> &a2, size_t size, string title, bool isprint);

void funcBoolAnd(RSSVectorBoolType &result, const RSSVectorBoolType &a1, const RSSVectorBoolType &a2, size_t size);

void funcBoolXor(RSSVectorBoolType &result, const RSSVectorBoolType &a1, const RSSVectorBoolType &a2, size_t size);

void funcBoolRev(vector<bool> &result, const RSSVectorBoolType &data, size_t size, string title, bool isprint);

template <typename Vec, typename T>
void funcB2AbyXOR(Vec &result, RSSVectorBoolType &data, size_t size, bool isbias);

template <typename Vec, typename T>
void funcB2A(Vec &result, const RSSVectorBoolType &data, size_t size, bool isbias);

// [b]^B = (b_0, b_1, b_2)
// P_A: (b_2, 0) (0, b_0)  (0, 0)
// P_B: (0, 0)   (b_0, 0)  (0, b_1)
// P_C: (0, b_2) (0, 0)    (b_1, 0)
template <typename Vec, typename T>
void funcB2AbyXOR(Vec &result, RSSVectorBoolType &data, size_t size, bool isbias)
{
	size_t float_precision = FLOAT_PRECISION;
	if (std::is_same<Vec, RSSVectorHighType>::value)
	{
		float_precision = HIGH_PRECISION;
	}
	else if (std::is_same<Vec, RSSVectorLowType>::value)
	{
		float_precision = LOW_PRECISION;
	}
	else
	{
		cout << "Not supported type" << typeid(data).name() << endl;
	}
	Vec aRss(size);
	Vec bRss(size);
	Vec amulb(size);
	T val = isbias ? FLOAT_BIAS : 1;
	vector<T> a(size);
	if (partyNum == PARTY_A)
	{

		for (size_t i = 0; i < size; i++)
		{
			a[i] = (data[i].first ^ data[i].second) ? val : 0;
		}

		funcPartySS<Vec, T>(aRss, a, size, PARTY_A);
		// funcShareSender<Vec, T>(aRss, a, size); // share a

		// TODO: precomute
		for (size_t i = 0; i < size; i++)
		{
			bRss[i] = make_pair(0, 0);
		}
	}
	else
	{
		funcPartySS<Vec, T>(aRss, a, size, PARTY_A);
		// funcShareReceiver<Vec, T>(aRss, size, PARTY_A); // get ss of a

		if (partyNum == PARTY_B)
		{
			for (size_t i = 0; i < size; i++)
			{
				bRss[i] = make_pair(0, data[i].second ? val : 0);
			}
		}
		else
		{
			for (size_t i = 0; i < size; i++)
			{
				bRss[i] = make_pair(data[i].first ? val : 0, 0);
			}
		}
	}

	// printRssVector<Vec>(aRss, "a_rss", size);
	// printRssVector<Vec>(bRss, "b_rss", size);

	// vector<T> a_plain(size);
	// vector<T> b_plain(size);

	// funcReconstruct<Vec, T>(aRss, a_plain, size, "a_plain", true);
	// funcReconstruct<Vec, T>(bRss, b_plain, size, "b_plain", true);

	// P_A: (a_2, a_0) (0,   0  )
	// P_B: (a_0, a_1) (0,   b_1)
	// P_C: (a_1, a_2) (b_1, 0  )

	// a ^ b1 = a + b1 - 2*a*b1
	if (isbias)
	{
		funcDotProduct(aRss, bRss, amulb, size, true, float_precision); // a * b1
	}
	else
	{
		funcDotProduct(aRss, bRss, amulb, size, false, float_precision); // a * b1
	}
	// printRssVector<Vec>(amulb, "a*b", size);

	// vector<T> amulb_plain(size);
	// funcReconstruct<Vec, T>(amulb, amulb_plain, size, "a*b_plain", true);

	for (size_t i = 0; i < size; i++)
	{
		result[i] = make_pair(aRss[i].first + bRss[i].first - 2 * amulb[i].first,
							  aRss[i].second + bRss[i].second - 2 * amulb[i].second);
	}
}

template <typename Vec, typename T>
void funcB2A(Vec &result, const RSSVectorBoolType &data, size_t size, bool isbias)
{
	RSSVectorBoolType randB(size);
	Vec randA(size);
	if (OFFLINE_ON)
	{
		PrecomputeObject->getB2ARand(randB, size);
		funcB2AbyXOR<Vec, T>(ref(randA), ref(randB), size, isbias);
	}
	// OFFLINE

	// vector<bool> rB(size);
	// funcBoolRev(rB, randB, size, "rand plain Bool", true);
	// vector<T> rA(size);
	// funcReconstruct<Vec, T>(randA, rA, size, "rand plain", true);
	// vector<bool> d(size);
	// funcBoolRev(d, data, size, "data plain Bool", true);

	RSSVectorBoolType cRss(size);
	funcBoolXor(cRss, data, randB, size); // mask data using rand
	vector<bool> c(size);
	funcBoolRev(c, cRss, size, "c plain", false);

	T val = isbias ? FLOAT_BIAS : 1;

	for (size_t i = 0; i < size; i++)
	{
		if (c[i])
		{ // x1 = 1-c
			if (partyNum == PARTY_B)
			{
				result[i] = make_pair(-randA[i].first, val - randA[i].second);
			}
			else if (partyNum == PARTY_C)
			{
				result[i] = make_pair(val - randA[i].first, -randA[i].second);
			}
			else
			{
				result[i] = make_pair(-randA[i].first, -randA[i].second);
			}
		}
		else
		{
			result[i] = make_pair(randA[i].first, randA[i].second);
		}
	}

	// vector<T> outputA(size);
	// funcReconstruct<Vec, T>(result, outputA, size, "output plain", true);

	// printRssVector(result, "b2a", size);
}

template <typename Vec, typename T>
void funcShareAB(Vec &result1, const vector<T> &data1, const size_t size1, RSSVectorBoolType &result2, const vector<bool> &data2, size_t size2, const int shareParty)
{
	PrecomputeObject->getZeroShareRand<Vec, T>(result1, size1, shareParty);
	PrecomputeObject->getZeroBShare(result2, size2, shareParty);
	if (partyNum == shareParty)
	{
		vector<T> a1_plus_data(size1);
		for (size_t i = 0; i < size1; i++)
		{
			T temp = data1[i] + result1[i].first;
			result1[i].first = temp;
			a1_plus_data[i] = temp;
		}

		// printBoolRssVec(result2, "result2", size2);
		vector<bool> a1_xor_data(size2);
		for (size_t i = 0; i < size2; i++)
		{
			bool temp = data2[i] ^ result2[i].first;
			result2[i].first = temp;
			a1_xor_data[i] = temp;
		}
		// printVector<T>(a1_plus_data, "a1+x", size1);
		// printBoolVec(a1_xor_data, "a1^x", size2);

		int plussize = ((size2 - 1) / sizeof(T)) + 1;
		vector<T> senddata(size1 + plussize);
		appendBool2u8(senddata, a1_plus_data, a1_xor_data, size1, size2); // a1_plus_data += a1_xor_data

		// printVector<T>(senddata, "send data", senddata.size());

		// bool2u8<T>(a1_xor_data, size2);
		// thread *threads = new thread[2];
		// send a1+x to prevparty
		// threads[0] = thread(sendBoolVector, ref(a1_xor_data), prevParty(partyNum), size2);
		// threads[1] = thread(sendVector<T>, ref(a1_plus_data), prevParty(partyNum), size1);
		sendVector<T>(senddata, prevParty(partyNum), size1 + plussize);
		// send a1^x to prevparty
		// cout << "send a1 xor data" << endl;
	}
	else if (partyNum == prevParty(shareParty))
	{
		// vector<T> a3A(size1);
		// PrecomputeObject->getZeroSharePrev<T>(a3A, size1);

		// vector<bool> a3B(size2);
		// PrecomputeObject->getZeroBSharePrev(a3B, size2);

		int plussize = ((size2 - 1) / sizeof(T)) + 1;
		vector<T> a1_plus_data(size1);
		vector<bool> a1_xor_data(size2);
		vector<T> receivedata(size1 + plussize);

		// thread *threads = new thread[2];
		// threads[0] = thread(receiveBoolVector, ref(a1_xor_data), shareParty, size2); // receive a1+x
		receiveVector<T>(receivedata, shareParty, size1 + plussize); // receive a1+x

		// printVector<T>(receivedata, "receive data", receivedata.size());
		// threads[0].join();
		// for (int i = 0; i < 2; ++i)
		// 	threads[i].join();
		splitu82Bool(receivedata, a1_plus_data, a1_xor_data, size1, size2);

		// printVector<T>(a1_plus_data, "a1+x", size1);
		// printBoolVec(a1_xor_data, "a1^x", size2);
		for (size_t i = 0; i < size1; i++)
		{
			result1[i].second = a1_plus_data[i];
		}

		for (size_t i = 0; i < size2; i++)
		{
			result2[i].second = a1_xor_data[i];
		}

		// merge2Vec<Vec, T>(result1, a3A, a1_plus_data, size1);
		// mergeBoolVec(result2, a3B, a1_xor_data, size2);
	}
}

// send data1(A share) and data2(B share)
template <typename Vec, typename T>
void funcShareABSender(Vec &result1, const vector<T> &data1, const size_t size1, RSSVectorBoolType &result2, const vector<bool> &data2, size_t size2)
{
	// assert(a.size() == size && "a.size mismatch for reconstruct function");

	PrecomputeObject->getZeroShareSender<Vec, T>(result1, size1);
	vector<T> a1_plus_data(size1);
	for (size_t i = 0; i < size1; i++)
	{
		T temp = data1[i] + result1[i].first;
		result1[i].first = temp;
		a1_plus_data[i] = temp;
	}

	PrecomputeObject->getZeroBShareSender(result2, size2);
	// printBoolRssVec(result2, "result2", size2);
	vector<bool> a1_xor_data(size2);
	for (size_t i = 0; i < size2; i++)
	{
		bool temp = data2[i] ^ result2[i].first;
		result2[i].first = temp;
		a1_xor_data[i] = temp;
	}
	// printVector<T>(a1_plus_data, "a1+x", size1);
	// printBoolVec(a1_xor_data, "a1^x", size2);

	int plussize = ((size2 - 1) / sizeof(T)) + 1;
	vector<T> senddata(size1 + plussize);
	appendBool2u8(senddata, a1_plus_data, a1_xor_data, size1, size2); // a1_plus_data += a1_xor_data

	// printVector<T>(senddata, "send data", senddata.size());

	// bool2u8<T>(a1_xor_data, size2);
	// thread *threads = new thread[2];
	// send a1+x to prevparty
	// threads[0] = thread(sendBoolVector, ref(a1_xor_data), prevParty(partyNum), size2);
	// threads[1] = thread(sendVector<T>, ref(a1_plus_data), prevParty(partyNum), size1);
	thread sender(sendVector<T>, ref(senddata), prevParty(partyNum), size1 + plussize);
	// send a1^x to prevparty
	// cout << "send a1 xor data" << endl;
	sender.join();

	// threads[0].join();

	// for (int i = 0; i < 2; ++i)
	// 	threads[i].join();
}

template <typename Vec, typename T>
void funcShareABReceiver(Vec &result1, const size_t size1, RSSVectorBoolType &result2, const size_t size2, const int shareParty)
{
	// assert(a.size() == size && "a.size mismatch for reconstruct function");

	if (partyNum == prevParty(shareParty))
	{
		vector<T> a3A(size1);
		PrecomputeObject->getZeroSharePrev<T>(a3A, size1);

		vector<bool> a3B(size2);
		PrecomputeObject->getZeroBSharePrev(a3B, size2);

		int plussize = ((size2 - 1) / sizeof(T)) + 1;
		vector<T> a1_plus_data(size1);
		vector<bool> a1_xor_data(size2);
		vector<T> receivedata(size1 + plussize);

		// thread *threads = new thread[2];
		// threads[0] = thread(receiveBoolVector, ref(a1_xor_data), shareParty, size2); // receive a1+x
		thread receiver(receiveVector<T>, ref(receivedata), shareParty, size1 + plussize); // receive a1+x
		// cout << "receive a1 xor data" << endl;
		receiver.join();

		// printVector<T>(receivedata, "receive data", receivedata.size());
		// threads[0].join();
		// for (int i = 0; i < 2; ++i)
		// 	threads[i].join();
		splitu82Bool(receivedata, a1_plus_data, a1_xor_data, size1, size2);

		// printVector<T>(a1_plus_data, "a1+x", size1);
		// printBoolVec(a1_xor_data, "a1^x", size2);

		merge2Vec<Vec, T>(result1, a3A, a1_plus_data, size1);
		mergeBoolVec(result2, a3B, a1_xor_data, size2);
	}
	else
	{
		PrecomputeObject->getZeroShareReceiver<Vec, T>(result1, size1);
		PrecomputeObject->getZeroBShareReceiver(result2, size2);
	}
}

// offline op
/**
 * @brief generate 0/1 random
 *
 * @tparam Vec
 * @tparam T
 * @param b
 * @param size
 */
template <typename Vec, typename T>
void funcRandBitByXor(Vec &b, size_t size)
{
	RSSVectorBoolType bRSS(size);
	PrecomputeObject->getBPairRand(bRSS, size);

	funcB2AbyXOR<Vec, T>(b, bRSS, size, false);
}