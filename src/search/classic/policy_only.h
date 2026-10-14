/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include <algorithm>

namespace lczero {
namespace classic {

// Return the minimum alternative prior as a fraction of the top prior, NOT
// the permitted drop. The permitted drop is (1+Q)/2, capped at 95%: zero at
// Q=-1, 50% at Q=0, 75% at Q=+0.5. The only flat regions are the endpoint
// safety clamps, never an intermediate band. The caller supplies value-head
// W-L from the side-to-move perspective; no additional evaluation is needed.
inline float PolicyOnlySafePriorRatio(float q) {
  return std::clamp(0.5f * (1.0f - q), 0.05f, 1.0f);
}

}  // namespace classic
}  // namespace lczero
