/*
* Functions for reading and processing txt file with floats
* Tomas Goldmann,2024
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

class Data {
public:
    size_t size;
    float *loc_x = NULL;
    float *loc_y = NULL;
    float *loc_w = NULL;
    float *loc_h = NULL;

    float *prior_x = NULL;
    float *prior_y = NULL;
    float *prior_w = NULL;
    float *prior_h = NULL;

    float *scores = NULL;

    Data(size_t size) : size(size) {
        loc_x = new float[size];
        loc_y = new float[size];
        loc_w = new float[size];
        loc_h = new float[size];

        prior_x = new float[size];
        prior_y = new float[size];
        prior_w = new float[size];
        prior_h = new float[size];

        scores = new float[size];
    }
};

vector<vector<float>> splitFloats(const vector<float>& floats, int index) ;
Data* readFloatsFromFile(const string& filename, vector<vector<float>>& priors, float threshold);