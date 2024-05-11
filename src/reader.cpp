/*
 * Function to read floats from txt
 * Tomas Goldmann,2023
 */

#include "reader.hpp"
#include <sys/mman.h> // Required for mmap
#include <unistd.h>   // Required for close
#include <cstring>
#include <x86intrin.h>
#include <immintrin.h>
#include <sys/stat.h>
#include <string.h>
#include "ipa_tool.h"
#include <fcntl.h>

using namespace std;

const char *split(const char *str, char delim, char *out = NULL)
{
    int i = 0;
    while (*str)
    {
        if (*str == delim)
        {
            str++;
            break;
        }
        out[i++] = *str++;
    }
    out[i] = '\0';
    return str;
}

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

Data *readFloatsFromFile(const string &filename, vector<vector<float>> &priors, float threshold)
{
    InstructionCounter counter;
    counter.start();

    auto fd = open(filename.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);

    if (fd == -1)
    {
        cerr << "Error: Unable to open file." << endl;
        return NULL; // or handle the error as you see fit
    }

    struct stat fileStat;
    auto yy = fstat(fd, &fileStat);
    if (yy < 0)
    {
        cerr << "Error: Unable to get file size." << endl;
        return NULL; // or handle the error as you see fit
    }

    auto file_size = fileStat.st_size;

    auto mapped_file = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (mapped_file == MAP_FAILED)
    {
        printf("ERror is %d", errno);
        cerr << "Error: Unable to map file." << endl;
        return NULL; // or handle the error as you see fit
    }

    auto content = static_cast<char *>(mapped_file);

    close(fd);

    printf("FILE MAPPED");
    counter.print();
    counter.end();
    counter.start();

    // auto content = static_cast<char *>(mapped_file);

    int *lines = new int[12601]{0};
    int loaded = 1;

    auto newline = _mm256_set1_epi8('\n');

    for (int i = 0; i < file_size; i += 32)
    {
        auto chunk = _mm256_loadu_si256(reinterpret_cast<__m256i *>(content + i));

        auto cmp = _mm256_cmpeq_epi8(chunk, newline);
        auto mask = _mm256_movemask_epi8(cmp);

        while (mask != 0)
        {
            auto tz = __builtin_ctz(mask);
            lines[loaded++] = i + tz + 1;
            mask &= ~(1 << tz);
        }
    }

    printf("LINES FOUND");
    counter.print();
    counter.end();
    counter.start();

    auto last_line = content + lines[loaded - 1];
    auto data = new Data(12600);
    auto buffer = new char[100];

    loaded = 0;
    for (int i = 0; i < 12600; i++)
    {
        counter.start();
        last_line = (char *)split(last_line, ',', buffer);
        last_line = (char *)split(last_line, ',', buffer);

        float f = stof(buffer);

        if (f <= threshold)
        {
            counter.end();
            printf("SKIPPED");
            counter.print();
            continue;
        }

        data->scores[loaded] = f;

        auto selected_line = content + lines[i];
        auto buffer2 = new char[100];
        for (int j = 0; j < 4; j++)
        {
            selected_line = (char *)split(selected_line, ',', buffer2);

            if (buffer2 == NULL)
            {
                break;
            }

            auto f2 = stof(buffer2);

            switch (j)
            {
            case 0:
                data->loc_x[loaded] = f2;
                break;
            case 1:
                data->loc_y[loaded] = f2;
                break;
            case 2:
                data->loc_w[loaded] = f2;
                break;
            case 3:
                data->loc_h[loaded] = f2;
                break;
            }
        }

        data->prior_x[loaded] = priors[i][0];
        data->prior_y[loaded] = priors[i][1];
        data->prior_w[loaded] = priors[i][2];
        data->prior_h[loaded] = priors[i][3];

        counter.end();
        printf("LOADED");
        counter.print();

        loaded++;
    }

    data->size = loaded;

    counter.end();
    printf("LINES FILTERED");
    counter.print();

    return data;
}

// Data *readFloatsFromFile(const string &filename, vector<vector<float>> &priors, float threshold)
// {
//     std::ifstream file(filename);
//     std::vector<std::string> lines;
//     std::string line;

//     InstructionCounter counter;
//     counter.start();
//     if (file.is_open())
//     {
//         while (std::getline(file, line))
//         {
//             lines.push_back(line);
//         }
//         file.close();
//         printf("LINES READ");
//         counter.print();
//         counter.end();
//         // counter.start();
//     }
//     else
//     {
//         std::cerr << "Error: Unable to open file." << std::endl;
//         return NULL; // or handle the error as you see fit
//     }

//     auto last_line = lines.back();
//     auto data = new Data(12600);

//     // Split the last line into a vector of floats
//     std::istringstream iss(last_line);
//     std::vector<float> floats;
//     int loaded = 0;
//     for (int i = 0; i < 12600; i++)
//     {
//         counter.start();
//         string out;
//         getline(iss, out, ',');
//         getline(iss, out, ',');

//         float f = stof(out);

//         if (f <= threshold)
//         {
//             continue;
//         }

//         data->scores[loaded] = f;

//         auto selected_line = lines[i];
//         // split the line into a vector of floats
//         // we will get 4 parts

//         std::istringstream iss2(selected_line);
//         for (int j = 0; j < 4; j++)
//         {
//             getline(iss2, out, ',');
//             float f2 = stof(out);

//             switch (j)
//             {
//             case 0:
//                 data->loc_x[loaded] = f2;
//                 break;
//             case 1:
//                 data->loc_y[loaded] = f2;
//                 break;
//             case 2:
//                 data->loc_w[loaded] = f2;
//                 break;
//             case 3:
//                 data->loc_h[loaded] = f2;
//                 break;
//             }
//         }

//         data->prior_x[loaded] = priors[i][0];
//         data->prior_y[loaded] = priors[i][1];
//         data->prior_w[loaded] = priors[i][2];
//         data->prior_h[loaded] = priors[i][3];

//         loaded++;
//         counter.end();
//         printf("LOOP ENDED");
//         counter.print();
//     }

//     data->size = loaded;

//     // printf("LINES PROCESSED");
//     // counter.print();
//     // counter.end();
//     // counter.start();

//     return data;
// }