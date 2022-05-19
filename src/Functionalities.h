
#pragma once
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <tuple>
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

/**
 * @brief generate ss of data and send to other party
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
	assert(a.size() == size && "a.size mismatch for reconstruct function");

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
	assert(a.size() == size && "a.size mismatch for reconstruct function");

	if (partyNum == prevParty(shareParty))
	{
		vector<T> a3(size);
		PrecomputeObject->getZeroSharePrev<T>(a3, size);

		vector<T> a1_plus_data(size);
		thread receiver(receiveVector<T>, ref(a1_plus_data), shareParty, size); // receive a1+x
		receiver.join();

		Merge2Vec<Vec, T>(a, a3, a1_plus_data, size);

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
			std::cout << str << ": \t\t";
			for (int i = 0; i < size; ++i)
				print_linear(b[i], "SIGNED");
			std::cout << std::endl;
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
	dividePlain(reconst, (1 << power));
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
		vector<computeType> temp_1(size, (1 << ((1 << j) + (int)alpha[0])));
		funcGetShares(temp, temp_1);
		subtractVectors<RSScomputeType>(x, temp, d, size);
		funcRELUPrime(d, c, size);
		funcReconstructBit(c, r_c, size, "null", false);
		if (r_c[0] == 0)
		{
			for (int i = 0; i < size; ++i)
				alpha[i] += (1 << j);
		}
	}
}

template <typename VEC>
void funcDivision(const VEC &a, const VEC &b, VEC &quotient,
				  size_t size)
{
	log_print("funcDivision");

	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, highBit, lowBit>::type computeType;
	typedef typename std::conditional<std::is_same<VEC, RSSVectorHighType>::value, RSSHighType, RSSLowType>::type RSScomputeType;

	// TODO incorporate funcPow
	// TODO Scale up and complete this computation with fixed-point precision
	vector<smallType> alpha_temp(size);
	funcPow(b, alpha_temp, size);

	size_t alpha = alpha_temp[0];
	size_t precision = alpha + 1;
	const computeType constTwoPointNine = ((computeType)(2.9142 * (1 << precision)));
	const computeType constOne = ((computeType)(1 * (1 << precision)));

	vector<computeType> data_twoPointNine(size, constTwoPointNine), data_one(size, constOne), reconst(size);
	VEC ones(size), twoPointNine(size), twoX(size), w0(size), xw0(size),
		epsilon0(size), epsilon1(size), termOne(size), termTwo(size), answer(size);
	funcGetShares(twoPointNine, data_twoPointNine);
	funcGetShares(ones, data_one);

	multiplyByScalar(b, 2, twoX);
	subtractVectors<RSScomputeType>(twoPointNine, twoX, w0, size);
	funcDotProduct(b, w0, xw0, size, true, precision);
	subtractVectors<RSScomputeType>(ones, xw0, epsilon0, size);
	if (PRECISE_DIVISION)
		funcDotProduct(epsilon0, epsilon0, epsilon1, size, true, precision);
	addVectors(ones, epsilon0, termOne, size);
	if (PRECISE_DIVISION)
		addVectors(ones, epsilon1, termTwo, size);
	funcDotProduct(w0, termOne, answer, size, true, precision);
	if (PRECISE_DIVISION)
		funcDotProduct(answer, termTwo, answer, size, true, precision);

	// RSSVectorMyType scaledA(size);
	// multiplyByScalar(a, (1 << (alpha + 1)), scaledA);
	funcDotProduct(answer, a, quotient, size, true, ((2 * precision - FLOAT_PRECISION)));
}

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
	funcDotProduct(b_repeat, a, quotient, batchSize * B, true, (2 * precision - FLOAT_PRECISION)); // Convert to fixed precision
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
	if (std::is_same<T, RSSVectorHighType>::value)
	{
		std::cout << "high" << std::endl;
	}
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
	else
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
		dividePlain(diffReconst, (1 << precision));
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

	dividePlain(diffReconst, (1 << truncation));

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
// void debugReduction();
// void debugPartySS();

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
template <typename Vec>
void print_vector(Vec &var, string type, string pre_text, int print_nos);
template <typename Vec>
void print_vector(Vec &var, string type, string pre_text, int print_nos)
{
	if (std::is_same<Vec, RSSVectorSmallType>::value)
	{
		size_t temp_size = var.size();
		vector<smallType> b(temp_size);
		funcReconstruct(var, b, temp_size, "anything", false);
		cout << pre_text << endl;
		for (int i = 0; i < print_nos; ++i)
		{
			cout << b[i] << " ";
			// if (i % 10 == 9)
			// std::cout << std::endl;
		}
		cout << endl;
	}
	else
	{
		size_t temp_size = var.size();
		typedef lowBit T;
		if (std::is_same<Vec, RSSVectorHighType>::value)
			typedef highBit T;
		vector<T> b(temp_size);
		funcReconstruct(var, b, temp_size, "anything", false);
		cout << pre_text << endl;
		for (int i = 0; i < print_nos; ++i)
		{
			print_linear(b[i], type);
			// if (i % 10 == 9)
			// std::cout << std::endl;
		}
		cout << endl;
	}
}
// void print_vector(RSSVectorLowType &var, string type, string pre_text, int print_nos);
// void print_vector(RSSVectorHighType &var, string type, string pre_text, int print_nos);

void print_vector(RSSVectorSmallType &var, string type, string pre_text, int print_nos);

/********************* Share Conversion Functionalites *********************/
void funcReduction(RSSVectorLowType &output, const RSSVectorHighType &input);
void funcExtension(RSSVectorHighType &output, const RSSVectorLowType &input);
void funcPosWrap(vector<highBit> &w, const RSSVectorLowType &input);
void funcMixedShareGeneration();
void funcTruncation(const RSSVectorHighType &a, const RSSVectorLowType &b, int trunc_bits);
void funcTruncAndReduce(const RSSVectorHighType &a, const RSSVectorLowType &b);

/********************* Mixed-Precision Activations Functionalites *********************/
void funcMReLU();