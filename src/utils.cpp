#include "utils.hpp"
#include <x86intrin.h>
#include <immintrin.h>

Data *nms(Data *bboxes, float threshold)
{
    auto threshold_v = _mm256_broadcast_ss(&threshold);

    std::vector<int> index(bboxes->size);
    for (size_t i = 0; i < index.size(); ++i)
    {
        index[i] = i;
    }

    std::sort(index.begin(), index.end(), [&](int a, int b)
              { return bboxes->scores[a] < bboxes->scores[b]; });

    for (size_t i = 0; i < index.size(); ++i)
    {
        if (i != index[i])
        {
            std::swap(bboxes->loc_x[i], bboxes->loc_x[index[i]]);
            std::swap(bboxes->loc_y[i], bboxes->loc_y[index[i]]);
            std::swap(bboxes->loc_w[i], bboxes->loc_w[index[i]]);
            std::swap(bboxes->loc_h[i], bboxes->loc_h[index[i]]);

            std::swap(bboxes->scores[i], bboxes->scores[index[i]]);
        }
    }

    int *removed = new int[bboxes->size] {0};
 
    for (int i = 0; i + 8 <= bboxes->size; i++)
    {
        auto current_xmin = _mm256_broadcast_ss(bboxes->loc_x + i);
        auto current_ymin = _mm256_broadcast_ss(bboxes->loc_y + i);
        auto current_xmax = _mm256_broadcast_ss(bboxes->loc_w + i);
        auto current_ymax = _mm256_broadcast_ss(bboxes->loc_h + i);

        auto current_w = _mm256_sub_ps(current_xmax, current_xmin);
        auto current_h = _mm256_sub_ps(current_ymax, current_ymin);

        auto current_are = current_w * current_h;

        for (size_t j = i + 1; j + 9 <= bboxes->size; j += 8)
        {
            // Load other boxes
            auto other_xmin = _mm256_loadu_ps(bboxes->loc_x + j);
            auto other_ymin = _mm256_loadu_ps(bboxes->loc_y + j);
            auto other_xmax = _mm256_loadu_ps(bboxes->loc_w + j);
            auto other_ymax = _mm256_loadu_ps(bboxes->loc_h + j);

            auto xmin = _mm256_max_ps(current_xmin, other_xmin);
            auto xmax = _mm256_min_ps(current_xmax, other_xmax);
            auto ymin = _mm256_max_ps(current_ymin, other_ymin);
            auto ymax = _mm256_min_ps(current_ymax, other_ymax);

            auto inter_area = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(xmax, xmin)) *
                              _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(ymax, ymin));

            auto other_area = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(other_xmax, other_xmin)) *
                              _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(other_ymax, other_ymin));

            auto union_area = current_are + other_area - inter_area;
            auto iou = inter_area / union_area;

            auto mask = _mm256_cmp_ps(iou, threshold_v, _CMP_GT_OQ);

            if (_mm256_testz_ps(mask, mask))
            {
                continue;
            }
            else
            {
                for (size_t k = 0; k < 8; k++)
                {
                    removed[j + k] = 1;
                }
            }
        }
    }

    // remove the boxes that are marked as removed
    auto new_data = new Data(12600);

    int added = 0;

    for (int i = 0; i < bboxes->size; i++)
    {
        if (removed[i] == 0)
        {
            new_data->loc_x[i] = bboxes->loc_x[i];
            new_data->loc_y[i] = bboxes->loc_y[i];
            new_data->loc_w[i] = bboxes->loc_w[i];
            new_data->loc_h[i] = bboxes->loc_h[i];
            added++;
        }
    }

    new_data->size = added;
    return new_data;
}

float calculate_box_area(float x, float y, float w, float h)
{
    return (w - x + 1) * (h - y + 1);
}

__m256 calculate_box_areas(__m256 x, __m256 y, __m256 w, __m256 h)
{
    auto x1 = _mm256_sub_ps(w, x);
    x1 = _mm256_add_ps(x1, _mm256_set1_ps(1.0f));
    auto y1 = _mm256_sub_ps(h, y);
    y1 = _mm256_add_ps(y1, _mm256_set1_ps(1.0f));
    return _mm256_mul_ps(x1, y1);
}