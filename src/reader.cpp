/*
 * Function to read floats from txt
 * Tomas Goldmann,2023
 */

#include "reader.hpp"

using namespace std;

vector<vector<float>> splitFloats(const vector<float> &floats, int index)
{

    vector<float> first_half;
    vector<float> second_half;

    // Ensure the original vector has enough elements for splitting
    if (floats.size() < 6 * 12600)
    {
        cerr << "Error: The input must containt 6*12600 floats" << endl;
        return {first_half, second_half}; // Return empty vectors
    }

    // Split the original vector into two separate vectors
    first_half.insert(first_half.end(), floats.begin(), floats.begin() + index);
    second_half.insert(second_half.end(), floats.begin() + index, floats.end());

    // Return a vector containing the two split vectors
    return {first_half, second_half};
}

Data* readFloatsFromFile(const string& filename, vector<vector<float>>& priors, float threshold)
{
    std::ifstream file(filename);
    std::vector<std::string> lines;
    std::string line;

    if (file.is_open())
    {
        while (std::getline(file, line))
        {
            lines.push_back(line);
        }
        file.close();
    }
    else
    {
        std::cerr << "Error: Unable to open file." << std::endl;
        return NULL; // or handle the error as you see fit
    }

    auto last_line = lines.back();
    auto data = new Data(12600);

    // Split the last line into a vector of floats
    std::istringstream iss(last_line);
    std::vector<float> floats;
    int loaded = 0;
    for (int i = 0; i < 12600; i++)
    {
        string out;
        getline(iss, out, ',');
        getline(iss, out, ',');

        float f = stof(out);

        if (f <= threshold)
        {
            continue;
        }

        loaded++;

        data->scores[i] = f;

        auto selected_line = lines[i];

        // split the line into a vector of floats
        std::istringstream iss2(selected_line);
        for (int j = 0; j < 4; j++)
        {
            float f2;
            iss2 >> f2;

            switch (j)
            {
            case 0:
                data->loc_x[i] = f2;
                break;
            case 1:
                data->loc_y[i] = f2;
                break;
            case 2:
                data->loc_w[i] = f2;
                break;
            case 3:
                data->loc_h[i] = f2;
                break;
            }
        }

        data->prior_x[i] = priors[i][0];
        data->prior_y[i] = priors[i][1];
        data->prior_w[i] = priors[i][2];
        data->prior_h[i] = priors[i][3];
    }

    data->size = loaded;

    return data;
}