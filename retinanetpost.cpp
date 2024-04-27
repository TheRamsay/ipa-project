
#include <cstring>
#include <string>
#include <iostream>
#include <inttypes.h>
#include "ipa_tool.h"

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
    // counter.start();
    /*******************Part to optmize*********************/

    auto data = readFloatsFromFile(argv[2]);

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

    // d->loc_x.resize(loaded);
    // counter.print();

    // vector<float> floats = readFloatsFromFile(argv[2]);
    // vector<vector<float>> split_vectors = splitFloats(floats, 12600*4);

    // vector<float> floatarr = split_vectors[0];
    // vector<float> floatarrscr = split_vectors[1];
    vector<float> floatarrscr = data->scores;

    // std::vector<std::vector<float>> loc_soa;
    // std::vector<std::vector<float>> loc;
    // for (size_t i = 0; i < num_anchors; i++) {
    //     loc.push_back({floatarr[i * 4], floatarr[i * 4 + 1], floatarr[i * 4 + 2], floatarr[i * 4 + 3]});
    // }

    // counter.start();
    counter.start();
    std::vector<std::vector<float>> ddecoded_boxes = decode(d, variances, CONFIDENCE_THRESHOLD);
    counter.print();
    // printf("decoded_boxes size %d\n", ddecoded_boxes[0].size());
    // counter.print();

    // ofstream fff;
    // fff.open("mlem_mine.txt");

    // for (size_t i = 0; i < 12600; i++) {
    // printf("%f %f %f %f\n", ddecoded_boxes[0][i], ddecoded_boxes[1][i], ddecoded_boxes[2][i], ddecoded_boxes[3][i]);
    // fff << ddecoded_boxes[0][i] << " " << ddecoded_boxes[1][i] << " " << ddecoded_boxes[2][i] << " " << ddecoded_boxes[3][i] << "\n";
    // }

    // fff.close();

    std::vector<std::vector<float>> decoded_boxes;

    for (size_t i = 0; i < loaded; i++)
    {
        decoded_boxes.push_back({ddecoded_boxes[0][i], ddecoded_boxes[1][i], ddecoded_boxes[2][i], ddecoded_boxes[3][i]});
    }

    // exit(0);

    // std::vector<std::vector<float>> decoded_boxes = decode(loc, priors, variances);
    // counter.print();

    std::vector<float> scores;
    std::vector<int> inds;
    std::vector<float> det_scores;

    std::vector<std::vector<float>> det_boxes;

    // printf("mLEML MLEMEL%d\n", floatarrscr.size());

    for (size_t i = 0; i < loaded; i++)
    {
        // scores.push_back(floatarrscr[i*2+1]);
        // if (floatarrscr[i] > CONFIDENCE_THRESHOLD)
        // {
            // inds.push_back(i);
            // decoded_boxes[i].push_back(floatarrscr[i*2+1]);
        decoded_boxes[i][0] = decoded_boxes[i][0] * 640;
        decoded_boxes[i][1] = decoded_boxes[i][1] * 480;
        decoded_boxes[i][2] = decoded_boxes[i][2] * 640;
        decoded_boxes[i][3] = decoded_boxes[i][3] * 480;
        det_boxes.push_back(decoded_boxes[i]);
            // det_scores.push_back(scores[i]);
        // }
    }

    auto out = nms(det_boxes, 0.4);

    // Test
    // f1(10);

    counter.print();

    counter.print();
    /************************************************/

    for (int i = 0; i < out.size(); i++)
    {
        // #ifdef DEBUG
        printf("Box %f %f %f %f %f\n", out[i][0], out[i][1], out[i][2], out[i][3], out[i][4]);
        // #endif

        cv::Rect roi((int)out[i][0], (int)out[i][1], (int)out[i][2] - (int)out[i][0], (int)out[i][3] - (int)out[i][1]);
        rectangle(image, roi, color, thickness);
    }

    imshow("Output", image);
    waitKey(0);

    return 0;
}