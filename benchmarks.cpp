#include <benchmark/benchmark.h>
#include "include/prior_boxes.hpp"
#include "include/reader.hpp"

#define CONFIDENCE_THRESHOLD 0.999
#define INPUT_WIDTH 640
#define INPUT_HEIGHT 480
#define ANCHORS_COUNT 12600

// Define another benchmark
static void Mlem(benchmark::State &state)
{
    std::vector<int> image_size = {INPUT_WIDTH, INPUT_HEIGHT};
    std::vector<float> variances = {0.1f, 0.2f};
    auto data = readFloatsFromFile("./input/input.txt");

    PriorBox priorBox(image_size, "projekt");

    std::vector<std::vector<float>> priors = priorBox.forward();

    // for (size_t i = 0; i < 12600; i++)
    // {
    //     data->prior_x.push_back(priors[i][0]);
    //     data->prior_y.push_back(priors[i][1]);
    //     data->prior_w.push_back(priors[i][2]);
    //     data->prior_h.push_back(priors[i][3]);
    // }


    Data *d = new Data();
    int loaded = 0;

    for (int i = 0; i < 12600; i++)
    {
        if (data->scores[i] <= CONFIDENCE_THRESHOLD)
        {
            continue;
        }

        loaded++;

        d->loc_x.push_back(data->loc_x[i]);
        d->loc_y.push_back(data->loc_y[i]);
        d->loc_w.push_back(data->loc_w[i]);
        d->loc_h.push_back(data->loc_h[i]);

        d->scores.push_back(data->scores[i]);

        d->prior_x.push_back(priors[i][0]);
        d->prior_y.push_back(priors[i][1]);
        d->prior_w.push_back(priors[i][2]);
        d->prior_h.push_back(priors[i][3]);

    }

    d->loc_x.resize(loaded);

    vector<float> floatarrscr = data->scores;
    for (auto _ : state)
        std::vector<std::vector<float>> ddecoded_boxes = decode(d, variances, CONFIDENCE_THRESHOLD);
}
BENCHMARK(Mlem);

BENCHMARK_MAIN();

// #include <benchmark/benchmark.h>
// #include "include/prior_boxes.hpp"
// #include "include/reader.hpp"

// #define CONFIDENCE_THRESHOLD 0.999
// #define INPUT_WIDTH 640
// #define INPUT_HEIGHT 480
// #define ANCHORS_COUNT 12600

// // Define another benchmark
// static void Mlem(benchmark::State &state)
// {
// 	std::vector<int> image_size = {INPUT_WIDTH, INPUT_HEIGHT};
// 	std::vector<float> variances = {0.1f, 0.2f};
//     auto data = readFloatsFromFile("./input/input.txt");

// 	PriorBox priorBox(image_size, "projekt");

//     std::vector<std::vector<float>>  priors = priorBox.forward();

//     for (size_t i = 0; i < 12600; i++)
//     {
//         data->prior_x.push_back(priors[i][0]);
//         data->prior_y.push_back(priors[i][1]);
//         data->prior_w.push_back(priors[i][2]);
//         data->prior_h.push_back(priors[i][3]);
//     }

//     vector<float> floatarrscr = data->scores;
//     for (auto _ : state)
//         std::vector<std::vector<float>> ddecoded_boxes = decode(data, variances, CONFIDENCE_THRESHOLD);
// }
// BENCHMARK(Mlem);

// BENCHMARK_MAIN();