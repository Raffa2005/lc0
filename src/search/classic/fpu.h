/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include <cmath>

namespace lczero {
namespace classic {

inline float ComputeFirstPlayUrgency(float configured_value, bool absolute,
                                     float parent_wl, float parent_d,
                                     float draw_score, float visited_policy,
                                     bool same_player) {
  if (absolute) return configured_value;

  // Node Q is stored from the previous mover's perspective. Ordinarily the
  // side to move is the opponent, but a mid-double node keeps the same mover.
  const float wl_sign = same_player ? 1.0f : -1.0f;
  const float side_to_move_q = wl_sign * parent_wl + draw_score * parent_d;
  return side_to_move_q -
         configured_value * std::sqrt(visited_policy);
}

}  // namespace classic
}  // namespace lczero
