/*
 * Prior boxes for RetinaNet
 * Tomas Goldmann,2023
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include "prior_boxes.hpp"
#include <x86intrin.h>
#include <immintrin.h>
#include <avx_mathfun.hpp>

PriorBox::PriorBox(std::vector<int> image_size, std::string phase)
{

    std::map<std::string, std::vector<std::vector<float>>> cfg;
    this->min_sizes = {{16.0f, 32.0f}, {64.0f, 128.0f}, {256.0f, 512.0f}};
    this->image_size = {image_size[1], image_size[0]};
    this->steps = {8, 16, 32};

    for (const float step : steps)
    {
        this->feature_maps.push_back({(int)std::ceil(image_size[0] / step), (int)std::ceil(image_size[1] / step)});
    }
}

std::vector<std::vector<float>> PriorBox::forward()
{

    std::vector<float> anchors;
    for (size_t k = 0; k < this->feature_maps.size(); k++)
    {
        const auto &f = this->feature_maps[k];
        const auto &min_sizes = this->min_sizes[k];

        for (int i = 0; i < f[1]; i++)
        {
            for (int j = 0; j < f[0]; j++)
            {
                for (const auto &min_size : min_sizes)
                {
                    float s_kx = min_size / this->image_size[1];
                    float s_ky = min_size / this->image_size[0];
                    std::vector<float> dense_cx = {static_cast<float>(j + 0.5) * this->steps[k] / this->image_size[1]};
                    std::vector<float> dense_cy = {static_cast<float>(i + 0.5) * this->steps[k] / this->image_size[0]};

                    for (const auto &cy : dense_cy)
                    {
                        for (const auto &cx : dense_cx)
                        {

                            anchors.push_back(cx);
                            anchors.push_back(cy);
                            anchors.push_back(s_kx);
                            anchors.push_back(s_ky);
                        }
                    }
                }
            }
        }
    }

    std::vector<std::vector<float>> output;

    size_t num_anchors = anchors.size() / 4;
    for (size_t i = 0; i < num_anchors; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            anchors[i * 4 + j] = std::min(std::max(anchors[i * 4 + j], 0.0f), 1.0f);
        }
        output.push_back({anchors[i * 4], anchors[i * 4 + 1], anchors[i * 4 + 2], anchors[i * 4 + 3]});
    }
    // std::cout << "len  " << output.size() << std::endl;

    return output;
}


std::vector<std::vector<float>> decode(Data *data, const std::vector<float> &variances, float confidence_threshold)
{
    std::vector<float> xmin(data->loc_x.size());
    std::vector<float> ymin(data->loc_x.size());
    std::vector<float> xmax(data->loc_x.size());
    std::vector<float> ymax(data->loc_x.size());

    __m256 var_x = _mm256_broadcast_ss(&variances[0]);
    __m256 var_y = _mm256_broadcast_ss(&variances[1]);
    __m256 var_w = _mm256_broadcast_ss(&variances[2]);
    __m256 var_h = _mm256_broadcast_ss(&variances[3]);

    __m256 threshold = _mm256_broadcast_ss(&confidence_threshold);

    std::vector<std::vector<float>> boxes;
    size_t bi = 0;

    for (size_t i = 0; i < data->loc_x.size(); i += 8)
    {
        if (i + 8 > data->loc_x.size())
        {
            break;
        }

        __m256 scores = _mm256_load_ps(&data->scores[i]);

        __m256 mask = _mm256_cmp_ps(scores, threshold, _CMP_GT_OQ);

        __m256 loc_x_r = _mm256_loadu_ps(&data->loc_x[i]);
        __m256 loc_y_r = _mm256_loadu_ps(&data->loc_y[i]);
        __m256 loc_w_r = _mm256_loadu_ps(&data->loc_w[i]);
        __m256 loc_h_r = _mm256_loadu_ps(&data->loc_h[i]);

        __m256 prior_x_r = _mm256_loadu_ps(&data->prior_x[i]);
        __m256 prior_y_r = _mm256_loadu_ps(&data->prior_y[i]);
        __m256 prior_w_r = _mm256_loadu_ps(&data->prior_w[i]);
        __m256 prior_h_r = _mm256_loadu_ps(&data->prior_h[i]);

        prior_w_r = _mm256_and_ps(prior_w_r, mask);
        prior_h_r = _mm256_and_ps(prior_h_r, mask);

        __m256 decode_x = _mm256_mul_ps(prior_w_r, var_x);
        decode_x = _mm256_mul_ps(decode_x, loc_x_r);
        decode_x = _mm256_add_ps(decode_x, prior_x_r);

        __m256 decode_y = _mm256_mul_ps(prior_h_r, var_y);
        decode_y = _mm256_mul_ps(decode_y, loc_y_r);
        decode_y = _mm256_add_ps(decode_y, prior_y_r);

        __m256 decode_w = _mm256_mul_ps(loc_w_r, var_w);
        decode_w = exp256_ps(decode_w);
        decode_w = _mm256_mul_ps(decode_w, prior_w_r);

        __m256 decode_h = _mm256_mul_ps(loc_h_r, var_h);
        decode_h = exp256_ps(decode_h);
        decode_h = _mm256_mul_ps(decode_h, prior_h_r);

        __m256 decode_xmin = _mm256_sub_ps(decode_x, _mm256_mul_ps(decode_w, _mm256_set1_ps(0.5f)));
        __m256 decode_ymin = _mm256_sub_ps(decode_y, _mm256_mul_ps(decode_h, _mm256_set1_ps(0.5f)));
        __m256 decode_xmax = _mm256_add_ps(decode_x, _mm256_mul_ps(decode_w, _mm256_set1_ps(0.5f)));
        __m256 decode_ymax = _mm256_add_ps(decode_y, _mm256_mul_ps(decode_h, _mm256_set1_ps(0.5f)));

        _mm256_storeu_ps(&xmin[i], decode_xmin);
        _mm256_storeu_ps(&ymin[i], decode_ymin);
        _mm256_storeu_ps(&xmax[i], decode_xmax);
        _mm256_storeu_ps(&ymax[i], decode_ymax);

    }

    return  { xmin, ymin, xmax, ymax };
}