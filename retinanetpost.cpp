
#include <cstring>
#include <string>
#include <iostream>
#include <inttypes.h>
#include "ipa_tool.h"

#include <sys/mman.h> // Required for mmap
#include <unistd.h>   // Required for close
#include <cstring>
#include <x86intrin.h>
#include <immintrin.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#include "prior_boxes.hpp"
#include "utils.hpp"
#include "reader.hpp"

// example: call extern function
extern "C"
{
    void f1(int a);
}

#define CONFIDENCE_THRESHOLD 0.999
#define INPUT_WIDTH 640
#define INPUT_HEIGHT 480
#define ANCHORS_COUNT 12600

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Run program by: ./retinapost input/vector.txt input/image.png";
    }

    Mat image = imread(argv[1]);

    if (image.empty())
    {
        cout << "Could not open or find a image" << endl;
        return -1;
    }

    // All constants refer to the configuration used in prior_boxes.cpp and to the 640x480 resolution.
    std::vector<int> image_size = {INPUT_WIDTH, INPUT_HEIGHT};
    std::vector<float> variances = {0.1f, 0.2f};
    size_t total0_len = ANCHORS_COUNT * 4;
    size_t total1_len = ANCHORS_COUNT * 2;
    size_t num_anchors = total0_len / 4;

    PriorBox priorBox(image_size, "projekt");

    std::vector<std::vector<float>> priors = priorBox.forward();

    Scalar color(0, 255, 0); // Color of the rectangle (in BGR)
    int thickness = 2;       // Thickness of the rectangle border

    InstructionCounter counter;
    counter.start();
    /*******************Part to optmize*********************/

    auto data = readFloatsFromFile(argv[2], priors, CONFIDENCE_THRESHOLD);

    if (data == NULL)
    {
        return -1;
    }

    auto floatarrscr = data->scores;
    printf("READER\t\t\t");
    counter.print();
    counter.end();
    counter.start();

    counter.start();

    auto ddecoded_boxes = decode(data, variances, CONFIDENCE_THRESHOLD);
    printf("DECODE\t\t\t");
    counter.print();
    counter.end();
    counter.start();

    std::vector<float> scores;
    std::vector<int> inds;
    std::vector<float> det_scores;

    std::vector<std::vector<float>> det_boxes;

    float w = 640.0f;
    float h = 480.0f;
    __m256 height = _mm256_broadcast_ss(&h);
    __m256 width = _mm256_broadcast_ss(&w);

    for (size_t i = 0; i < data->size; i += 8)
    {
        if (i + 8 > data->size)
        {
            break;
        }

        __m256 ddecoded_boxes_x = _mm256_loadu_ps(&ddecoded_boxes[0][i]);
        __m256 ddecoded_boxes_y = _mm256_loadu_ps(&ddecoded_boxes[1][i]);
        __m256 ddecoded_boxes_w = _mm256_loadu_ps(&ddecoded_boxes[2][i]);
        __m256 ddecoded_boxes_h = _mm256_loadu_ps(&ddecoded_boxes[3][i]);

        ddecoded_boxes_x = _mm256_mul_ps(ddecoded_boxes_x, width);
        ddecoded_boxes_y = _mm256_mul_ps(ddecoded_boxes_y, height);
        ddecoded_boxes_w = _mm256_mul_ps(ddecoded_boxes_w, width);
        ddecoded_boxes_h = _mm256_mul_ps(ddecoded_boxes_h, height);

        _mm256_storeu_ps(&(ddecoded_boxes[0][i]), ddecoded_boxes_x);
        _mm256_storeu_ps(&(ddecoded_boxes[1][i]), ddecoded_boxes_y);
        _mm256_storeu_ps(&(ddecoded_boxes[2][i]), ddecoded_boxes_w);
        _mm256_storeu_ps(&(ddecoded_boxes[3][i]), ddecoded_boxes_h);
    }

    printf("BEFORE NMS\t\t\t");
    counter.print();
    counter.end();
    counter.start();

    auto mlem_data = new Data(data->size);
    mlem_data->loc_x = ddecoded_boxes[0];
    mlem_data->loc_y = ddecoded_boxes[1];
    mlem_data->loc_w = ddecoded_boxes[2];
    mlem_data->loc_h = ddecoded_boxes[3];
    mlem_data->scores = data->scores;

    auto out = nms(mlem_data, 0.4);

    printf("NMS\t\t\t");
    counter.print();
    counter.end();

    // Test
    // f1(10);

    // counter.print();

    // counter.print();
    // /************************************************/

    for (int i = 0; i < out->size; i++)
    {
#ifdef DEBUG
        // printf("Box %f %f %f %f %f\n", out[i][0], out[i][1], out[i][2], out[i][3], out[i][4]);
        printf("Box %f %f %f %f\n", out->loc_x[i], out->loc_y[i], out->loc_w[i], out->loc_y[i]);
#endif

        cv::Rect roi((int)out->loc_x[i], (int)out->loc_y[i], (int)out->loc_w[i] - (int)out->loc_x[i], (int)out->loc_h[i] - (int)out->loc_y[i]);
        rectangle(image, roi, color, thickness);
    }

    imshow("Output", image);
    waitKey(0);

    return 0;
}

