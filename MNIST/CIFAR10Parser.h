#pragma once
#include "globals.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

#define DATASET "CIFAR10"
#define TRAINING_DATA_SIZE 50000
#define TEST_DATA_SIZE 10000
#define INPUT_DIMENSION 32*32*3
#define OUTPUT_DIMENSION 10

void ERROR(string str)
{
	cout << "ERROR: " << str << endl;
	exit(1); 
}

int parse(char* filename, string type)
{
	ifstream input_file(filename);
	ofstream data_file_A;
	ofstream data_file_B;
	ofstream data_file_C;
	ofstream label_file_A;
	ofstream label_file_B;
	ofstream label_file_C;
	
	string line;
	int data_size;
	int *inputs;

	if (type == TRAINING)
	{
		data_size = TRAINING_DATA_SIZE;
		data_file_A.open(TRAINING_DATA_A "_" DATASET);
		data_file_B.open(TRAINING_DATA_B "_" DATASET);
		data_file_C.open(TRAINING_DATA_C "_" DATASET);
		label_file_A.open(TRAINING_LABEL_A "_" DATASET);
		label_file_B.open(TRAINING_LABEL_B "_" DATASET);
		label_file_C.open(TRAINING_LABEL_C "_" DATASET);
	}
    // FIXME: CIFAR10 does not have labels for test data
	else if (type == TESTING)
	{
		data_size = TEST_DATA_SIZE;
		data_file_A.open(TESTING_DATA_A "_" DATASET);
		data_file_B.open(TESTING_DATA_B "_" DATASET);
		data_file_C.open(TESTING_DATA_C "_" DATASET);
		label_file_A.open(TESTING_LABEL_A "_" DATASET);
		label_file_B.open(TESTING_LABEL_B "_" DATASET);
		label_file_C.open(TESTING_LABEL_C "_" DATASET);
	}
	else
		ERROR("parse() only accepts TRAINING/TESTING");
	
    // jump the first row, the attributes info
	getline(input_file, line);
    cout << "Index: " << line << endl;

	//Reading
	inputs = new int [data_size * (INPUT_DIMENSION + 1)];

	for (int row = 0; row < data_size; ++row)
	{
		getline(input_file, line);
		// if (row == 0)
		// 	continue;
		stringstream lineStream(line);
		string cell;
								
		for (int column = 0; column < INPUT_DIMENSION + 1; ++column)
		{
			getline(lineStream, cell, ',');

			// if (!cell.empty())
			inputs[row*(INPUT_DIMENSION+1) + column] = stoi(cell);
		}
	}

	//Writing
	for (int row = 0; row < data_size; ++row)
	{
		for (int column = 0; column < INPUT_DIMENSION; ++column)
		{
			data_file_A << inputs[row*(INPUT_DIMENSION+1) + column] << "\t";
			data_file_B << 0 << "\t";
			data_file_C << 0 << "\t";
		}
		data_file_A << endl;
		data_file_B << endl;
		data_file_C << endl;
	}

	for (int row = 0; row < data_size; ++row)
	{
		for (int column = 0; column < OUTPUT_DIMENSION; ++column)
		{
			if (column == inputs[row*(INPUT_DIMENSION + 1) + INPUT_DIMENSION])
				label_file_A << 1 << "\t";
			else
				label_file_A << 0 << "\t";
			label_file_B << 0 << "\t";
			label_file_C << 0 << "\t";
		}
		label_file_A << endl;
		label_file_B << endl;
		label_file_C << endl;
	}

	delete[] inputs;
	input_file.close();
	data_file_A.close();
	data_file_B.close();
	data_file_C.close();
	label_file_A.close();
	label_file_B.close();
	label_file_C.close();

	return 0;
}