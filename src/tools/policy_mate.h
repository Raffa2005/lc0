// Restricted exhaustive White mate search against a fixed Black policy.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <chrono>
#include <functional>
#include <ostream>
#include <span>
#include <vector>

#include "chess/gamestate.h"

namespace lczero {

struct PolicyMateOptions {
  int white_turns = 4;
  int threads = 1;
  int batch_size = 256;
  int seconds = 600;
  size_t max_frontier = 1000000;
  int black_visits = 0;
};

struct PolicyMateExpansion {
  uint64_t double_pairs = 0;
  uint64_t ordinary_moves = 0;
  std::vector<GameState> mates;
  std::vector<GameState> black_positions;
};

// No NN work: all ordinary moves and all legal two-move mating finishes.
PolicyMateExpansion ExpandPolicyMateWhite(const GameState& state,
                                         bool keep_continuations);
std::string PolicyMateUci(const GameState& state);
size_t PolicyMateBest(std::span<const float> priors, const MoveList& legal);

struct PolicyMateDecision {
  std::vector<float> priors;
  size_t selected;
  int64_t visits = 0;
};

using PolicyMateEvaluator = std::function<std::vector<PolicyMateDecision>(
    std::span<const GameState>, const std::vector<MoveList>&)>;

// JSONL progress/results plus a complete TSV trace of Black's policy choices.
void SearchPolicyMates(const GameState& start, const PolicyMateOptions& opts,
                      const PolicyMateEvaluator& evaluate, std::ostream& output,
                      std::ostream& trace);
void PolicyMateCommand();

}  // namespace lczero
