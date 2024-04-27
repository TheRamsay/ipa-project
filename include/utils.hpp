/*
* Non-maximum suppression
* Tomas Goldmann,2024
*/

#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<std::vector<float>> nms(std::vector<std::vector<float>>& bboxes, float threshold);


#endif // UTILS_HPP
