/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

namespace lczero {
namespace classic {

// Policy-only repetition aversion keeps the existing 50% prior band while
// the side-to-move value is at most +0.5. Above that, confidence in the win
// widens the band linearly until a safe move may have 5% of the top prior at
// Q=+1. The caller supplies the value-head W-L score from the side-to-move
// perspective.
inline float PolicyOnlySafePriorRatio(float q) {
  if (q <= 0.5f) return 0.5f;
  if (q >= 1.0f) return 0.05f;
  return 0.95f - 0.9f * q;
}

}  // namespace classic
}  // namespace lczero
