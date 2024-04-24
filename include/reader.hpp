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


vector<vector<float>> splitFloats(const vector<float>& floats, int index) ;
vector<float> readFloatsFromFile(const string& filename);