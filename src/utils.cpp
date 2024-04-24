#include "utils.hpp"


bool CompareBBox(const std::vector<float> & a, const std::vector<float> & b)
{
    return a[4] > b[4];
}



std::vector<std::vector<float>> nms(std::vector<std::vector<float>>& bboxes, float threshold)
{
    std::vector<std::vector<float>> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        std::vector<float> select_bbox = bboxes[select_idx];
        float area1 = static_cast<float>((select_bbox[2] - select_bbox[0] + 1) * (select_bbox[3] - select_bbox[1] + 1));
        float x1 = static_cast<float>(select_bbox[0]);
        float y1 = static_cast<float>(select_bbox[1]);
        float x2 = static_cast<float>(select_bbox[2]);
        float y2 = static_cast<float>(select_bbox[3]);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            std::vector<float>& bbox_i = bboxes[i];
            float x = std::max<float>(x1, static_cast<float>(bbox_i[0]));
            float y = std::max<float>(y1, static_cast<float>(bbox_i[1]));
            float w = std::min<float>(x2, static_cast<float>(bbox_i[2])) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i[3])) - y + 1;
            
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i[2] - bbox_i[0] + 1) * (bbox_i[3] - bbox_i[1] + 1));
            float area_intersect = w * h;

   
            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}