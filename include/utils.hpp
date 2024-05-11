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
#include <reader.hpp>

Data * nms(Data *bboxes, float threshold);


#endif // UTILS_HPP
