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



Data* readFloatsFromFile(const string& filename) {
    vector<float> floats;
    ifstream file(filename);

    // Check if the file is opened successfully
    if (!file.is_open()) {
        cerr << "Error opening the file" << endl;
        // return floats; // Return an empty vector if the file cannot be opened
        return NULL; // Return an empty vector if the file cannot be opened
    }

    Data *data = new Data();
    string line;

    // Read each line from the file
    int line_index = 0;
    while (getline(file, line)) {
        // Create a string stream from the line
        stringstream ss(line);
        string token;

        // Split the line by commas and read each float
        // while (getline(ss, token, ',')) {
        // // while (getline(ss, token, ',')) {
        //     // Convert the string to a float and store it in the vector
        //     floats.push_back(stof(token));
        // }

        if (line_index++ == 12600) {

            int mlem = 0;
            while (getline(ss, token, ',')) {
                if (mlem++ % 2 != 0) {
                    data->scores.push_back(stof(token));
                }
            }
            break;
        }

        auto vectors = {
            data->loc_x,
            data->loc_y,
            data->loc_w,
            data->loc_h
        };

        int ii = 0;
        for (auto &vec : vectors) {
            getline(ss, token, ',');
            switch (ii) {
                case 0:
                    data->loc_x.push_back(stof(token));
                    break;
                case 1:
                    data->loc_y.push_back(stof(token));
                    break;
                case 2:
                    data->loc_w.push_back(stof(token));
                    break;
                case 3:
                    data->loc_h.push_back(stof(token));
                    break;
            }

            ii++;
            // vec.push_back(1.0);
            // vec.push_back(stof(token));
        }
    }

    // Close the file
    file.close();

    return data;
    // return floats;
}