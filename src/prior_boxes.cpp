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

std::vector<std::vector<float>> decode(const std::vector<std::vector<float>> &loc, const std::vector<std::vector<float>> &priors, const std::vector<float> &variances)
{
    // typedef float v8f __attribute__((vector_size(32)));

    float loc_x[loc.size()] = {};
    float loc_y[loc.size()] = {};
    float loc_w[loc.size()] = {};
    float loc_h[loc.size()] = {};

    float prior_x[priors.size()] = {};
    float prior_y[priors.size()] = {};
    float prior_w[priors.size()] = {};
    float prior_h[priors.size()] = {};

    std::vector<float> xmin(loc.size());
    std::vector<float> ymin(loc.size());
    std::vector<float> xmax(loc.size());
    std::vector<float> ymax(loc.size());

    for (size_t i = 0; i < loc.size(); i++)
    {
        loc_x[i] = loc[i][0];
        loc_y[i] = loc[i][1];
        loc_w[i] = loc[i][2];
        loc_h[i] = loc[i][3];

        prior_x[i] = priors[i][0];
        prior_y[i] = priors[i][1];
        prior_w[i] = priors[i][2];
        prior_h[i] = priors[i][3];
    }

    __m256 var_x = _mm256_broadcast_ss(&variances[0]);
    __m256 var_y = _mm256_broadcast_ss(&variances[1]);
    __m256 var_w = _mm256_broadcast_ss(&variances[2]);
    __m256 var_h = _mm256_broadcast_ss(&variances[3]);

    std::vector<std::vector<float>> boxes;
    size_t bi = 0;
    for (size_t i = 0; i < loc.size(); i += 8)
    {
        // We load two elements at a time
        __m256 loc_x_r = _mm256_loadu_ps(&loc_x[i]);
        __m256 loc_y_r = _mm256_loadu_ps(&loc_y[i]);
        __m256 loc_w_r = _mm256_loadu_ps(&loc_w[i]);
        __m256 loc_h_r = _mm256_loadu_ps(&loc_h[i]);

        __m256 prior_x_r = _mm256_loadu_ps(&prior_x[i]);
        __m256 prior_y_r = _mm256_loadu_ps(&prior_y[i]);
        __m256 prior_w_r = _mm256_loadu_ps(&prior_w[i]);
        __m256 prior_h_r = _mm256_loadu_ps(&prior_h[i]);

        // loc   [x, y, w, h, x, y, w, h]
        // prior [x, y, w, h, x, y, w, h]
        // var   [x, y, w, h, x, y, w, h]

        // decoded

        // const auto &prior = priors[i];
        // const auto &loc_pred = loc[i];
        // float prior_x = prior[0];
        // float prior_y = prior[1];
        // float prior_w = prior[2];
        // float prior_h = prior[3];
        // float var_x = variances[0];
        // float var_y = variances[1];
        // float var_w = variances[2];
        // float var_h = variances[3];

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

        // boxes.push_back({decode_xmin[0], decode_ymin[0], decode_xmax[0], decode_ymax[0]});
        // boxes.push_back({decode_xmin[1], decode_ymin[1], decode_xmax[1], decode_ymax[1]});
        // boxes.push_back({decode_xmin[2], decode_ymin[2], decode_xmax[2], decode_ymax[2]});
        // boxes.push_back({decode_xmin[3], decode_ymin[3], decode_xmax[3], decode_ymax[3]});
        // boxes.push_back({decode_xmin[4], decode_ymin[4], decode_xmax[4], decode_ymax[4]});
        // boxes.push_back({decode_xmin[5], decode_ymin[5], decode_xmax[5], decode_ymax[5]});
        // boxes.push_back({decode_xmin[6], decode_ymin[6], decode_xmax[6], decode_ymax[6]});
        // boxes.push_back({decode_xmin[7], decode_ymin[7], decode_xmax[7], decode_ymax[7]});

        // for (size_t j = 0; j < 8; j++)
        // {
        //     boxes.push_back({decode_xmin[j], decode_ymin[j], decode_xmax[j], decode_ymax[j]});
        // }

        // float decoded_x = prior_x + loc_pred[0] * var_x * prior_w;
        // float decoded_y = prior_y + loc_pred[1] * var_y * prior_h;
        // float decoded_w = prior_w * std::exp(loc_pred[2] * var_w);
        // float decoded_h = prior_h * std::exp(loc_pred[3] * var_h);

        // float decoded_xmin = decoded_x - decoded_w / 2;
        // float decoded_ymin = decoded_y - decoded_h / 2;
        // float decoded_xmax = decoded_x + decoded_w / 2;
        // float decoded_ymax = decoded_y + decoded_h / 2;

        // boxes.push_back({decoded_xmin, decoded_ymin, decoded_xmax, decoded_ymax});
    }

    // for (size_t i = 0; i < loc.size(); i++)
    // {
    //     boxes.push_back({xmin[i], ymin[i], xmax[i], ymax[i]});
    // }

    // return boxes;
    return  { xmin, ymin, xmax, ymax };
}