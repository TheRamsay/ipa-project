/*
* Function to read floats from txt
* Tomas Goldmann,2023
*/


#include "reader.hpp"

using namespace std;

vector<vector<float>> splitFloats(const vector<float>& floats, int index) {

    vector<float> first_half;
    vector<float> second_half;

    // Ensure the original vector has enough elements for splitting
    if (floats.size() < 6*12600) {
        cerr << "Error: The input must containt 6*12600 floats" << endl;
        return {first_half, second_half}; // Return empty vectors
    }

    // Split the original vector into two separate vectors
    first_half.insert(first_half.end(), floats.begin(), floats.begin() + index);
    second_half.insert(second_half.end(), floats.begin() + index,  floats.end() );

    // Return a vector containing the two split vectors
    return {first_half, second_half};
}



vector<float> readFloatsFromFile(const string& filename) {
    vector<float> floats;
    ifstream file(filename);

    // Check if the file is opened successfully
    if (!file.is_open()) {
        cerr << "Error opening the file" << endl;
        return floats; // Return an empty vector if the file cannot be opened
    }

    string line;

    // Read each line from the file
    while (getline(file, line)) {
        // Create a string stream from the line
        stringstream ss(line);
        string token;

        // Split the line by commas and read each float
        while (getline(ss, token, ',')) {
            // Convert the string to a float and store it in the vector
            floats.push_back(stof(token));
        }
    }

    // Close the file
    file.close();

    return floats;
}