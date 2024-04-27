/*
* Prior boxes for RetinaNet
* Tomas Goldmann,2023
*/

#ifndef PRIOR_BOXES_HPP
#define PRIOR_BOXES_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include "reader.hpp"

class PriorBox {


private:
    std::vector<std::vector<int>> feature_maps;
    std::vector<std::vector<float>> min_sizes;
    std::vector<int> steps;
    bool clip;
    std::vector<int> image_size;
    std::string name;


public:
    PriorBox(std::vector<int> image_size = std::vector<int>(), std::string phase = "train");
    std::vector<std::vector<float>> forward();

};

std::vector<std::vector<float>> decode(Data *data, const std::vector<float>& variances, float confidence_threshold);
// std::vector<std::vector<float>> decode(const std::vector<std::vector<float>>& loc, const std::vector<std::vector<float>>& priors, const std::vector<float>& variances);


#endif // PRIOR_BOXES_HPP