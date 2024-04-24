/*
* Non-maximum suppression
* Tomas Goldmann,2024
*/


#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<std::vector<float>> nms(std::vector<std::vector<float>>& bboxes, float threshold);
