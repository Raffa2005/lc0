// SPDX-License-Identifier: GPL-3.0-or-later
#include "tools/policy_mate.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>

#include "search/classic/node.h"
#include "utils/exception.h"

namespace lczero {
namespace {
using Clock = std::chrono::steady_clock;

GameState WithMoves(const GameState& state, std::initializer_list<Move> moves) {
  auto result = state;
  result.moves.insert(result.moves.end(), moves.begin(), moves.end());
  return result;
}

double Elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
}  // namespace

std::string PolicyMateUci(const GameState& state) {
  auto history = state.GetPositionHistory();
  std::ostringstream result;
  for (size_t i = 0; i < state.moves.size(); ++i) {
    if (i) result << ' ';
    auto move = state.moves[i];
    if (history.GetPositionAt(i).IsBlackToMove()) move.Flip();
    result << move.ToString(false);
  }
  return result.str();
}

size_t PolicyMateBest(std::span<const float> priors, const MoveList& legal) {
  if (priors.empty() || priors.size() != legal.size()) throw Exception("Invalid policy size");
  for (size_t i = 0; i < priors.size(); ++i) {
    if (!std::isfinite(priors[i]) || priors[i] < 0 || priors[i] > 1) {
      throw Exception("Invalid policy probability");
    }
  }
  // Reuse classic's compression AND edge sort (which is not stable on ties).
  // At zero child visits its subsequent partial_sort(count=1) keeps this top.
  classic::Node node(nullptr, 0);
  node.CreateEdges(legal);
  size_t i = 0;
  for (auto& edge : node.Edges()) edge.edge()->SetP(priors[i++]);
  node.SortEdges();
  for (auto& edge : node.Edges()) {
    return std::find(legal.begin(), legal.end(), edge.GetMove()) - legal.begin();
  }
  throw Exception("Missing policy edge");
}

PolicyMateExpansion ExpandPolicyMateWhite(const GameState& state,
                                         bool keep_continuations) {
  PolicyMateExpansion result;
  auto history = state.GetPositionHistory();
  if (history.IsBlackToMove() || history.Last().IsMidDoublemove() ||
      !history.Last().OurDoublemoveAvailable() ||
      !history.Last().TheirDoublemoveAvailable()) {
    throw Exception("Expected White turn with both abilities retained");
  }
  if (history.ComputeGameResult() != GameResult::UNDECIDED) return result;
  const auto moves = GenerateDoubleMoveBoardMoves(history.Last());
  history.Reserve(history.GetLength() + 2);
  for (Move first : moves) {
    ++result.ordinary_moves;
    history.Append(first);
    const auto ordinary_result = history.ComputeGameResult();
    if (ordinary_result == GameResult::WHITE_WON) {
      result.mates.push_back(WithMoves(state, {first}));
    } else if (keep_continuations && ordinary_result == GameResult::UNDECIDED) {
      result.black_positions.push_back(WithMoves(state, {first}));
    }
    history.Pop();

    history.Append(first, /*is_first_of_double=*/true);
    // Generating continuations itself checks completable activation; the first
    // half is not a completed turn and is never tested for terminal mate.
    for (Move second : GenerateDoubleMoveBoardMoves(history.Last())) {
      ++result.double_pairs;
      history.Append(second);
      // Checking the king first avoids full reply generation on non-checks.
      if (history.Last().GetBoard().IsUnderCheck() &&
          history.ComputeGameResult() == GameResult::WHITE_WON) {
        result.mates.push_back(
            WithMoves(state, {first.WithAbilityActivation(), second}));
      }
      history.Pop();
    }
    history.Pop();
  }
  return result;
}

void SearchPolicyMates(const GameState& start, const PolicyMateOptions& opts,
                      const PolicyMateEvaluator& evaluate, std::ostream& output,
                      std::ostream& trace) {
  if (opts.white_turns < 1 || opts.threads < 1 || opts.batch_size < 1 ||
      opts.seconds < 1 || opts.max_frontier < 1) {
    throw Exception("Invalid policy mate search limits");
  }
  const auto started = Clock::now();
  const auto deadline = started + std::chrono::seconds(opts.seconds);
  std::vector<GameState> frontier{start};
  int complete_depth = 0;
  uint64_t queries = 0, pairs = 0, pruned = 0;
  std::string status = "depth_limit";
  trace << "white_turn\tprefix_uci\tblack_move\tprior\trunner_up_prior\tpruned_ability\tvisits\n";
  output << "{\"type\":\"config\",\"white_turns\":" << opts.white_turns
         << ",\"batch_size\":" << opts.batch_size << ",\"threads\":" << opts.threads
         << ",\"seconds\":" << opts.seconds << ",\"max_frontier\":" << opts.max_frontier
         << ",\"black_visits\":" << opts.black_visits
         << ",\"black_policy\":\""
         << (opts.black_visits ? "classic_mcts" : "classic_raw_prior_argmax") << "\","
            "\"repetition_override\":false,\"prune_black_activation\":true,"
            "\"white_ability_only_at_mating_finish\":true}" << std::endl;
  for (int depth = 1; depth <= opts.white_turns; ++depth) {
    const auto depth_started = Clock::now();
    std::vector<PolicyMateExpansion> expanded(frontier.size());
    std::atomic<size_t> cursor{0}, done{0};
    std::atomic<size_t> pending_black{0};
    std::atomic<bool> frontier_limit{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto worker = [&] {
      try {
        while (Clock::now() < deadline) {
          const auto i = cursor.fetch_add(1);
          if (i >= frontier.size()) break;
          expanded[i] = ExpandPolicyMateWhite(
              frontier[i], depth < opts.white_turns && !frontier_limit);
          const auto count = expanded[i].black_positions.size();
          if (pending_black.fetch_add(count) + count > opts.max_frontier) {
            frontier_limit = true;
            expanded[i].black_positions.clear();
          }
          ++done;
        }
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (!error) error = std::current_exception();
      }
    };
    std::vector<std::thread> workers;
    const size_t threads = std::min<size_t>(opts.threads, frontier.size());
    for (size_t t = 1; t < threads; ++t) workers.emplace_back(worker);
    worker();
    for (auto& thread : workers) thread.join();
    if (error) std::rethrow_exception(error);
    uint64_t depth_pairs = 0, ordinary = 0, mates = 0;
    size_t candidate_count = 0;
    for (const auto& entry : expanded) candidate_count += entry.black_positions.size();
    std::vector<GameState> black;
    if (!frontier_limit) black.reserve(candidate_count);
    for (auto& entry : expanded) {
      depth_pairs += entry.double_pairs;
      ordinary += entry.ordinary_moves;
      for (const auto& mate : entry.mates) {
        ++mates;
        output << "{\"type\":\"mate\",\"white_turn\":" << depth
               << ",\"moves_uci\":\"" << PolicyMateUci(mate)
               << "\",\"fen\":\"" << PositionToFen(mate.CurrentPosition())
               << "\"}" << '\n';
      }
      if (!frontier_limit) {
        std::move(entry.black_positions.begin(), entry.black_positions.end(),
                  std::back_inserter(black));
      }
    }
    pairs += depth_pairs;
    const bool complete = done == frontier.size();
    if (complete) complete_depth = depth;
    output << "{\"type\":\"depth\",\"white_turn\":" << depth
           << ",\"complete\":" << (complete ? "true" : "false")
           << ",\"white_positions\":" << frontier.size()
           << ",\"white_positions_checked\":" << done
           << ",\"ordinary_moves\":" << ordinary << ",\"double_pairs\":" << depth_pairs
           << ",\"mates\":" << mates << ",\"cpu_phase_seconds\":" << Elapsed(depth_started)
           << "}" << std::endl;
    if (!complete) { status = "time_limit"; break; }
    if (mates) { status = "shortest_found_within_restrictions"; break; }
    if (depth == opts.white_turns) break;
    if (frontier_limit) { status = "frontier_limit"; break; }
    expanded.clear();
    frontier.clear();
    const auto inference_started = Clock::now();
    uint64_t depth_pruned = 0, terminal = 0, depth_queries = 0;
    for (size_t offset = 0; offset < black.size(); offset += opts.batch_size) {
      if (Clock::now() >= deadline) { status = "time_limit"; break; }
      auto batch = std::span<const GameState>(black).subspan(
          offset, std::min<size_t>(opts.batch_size, black.size() - offset));
      std::vector<MoveList> legal;
      for (const auto& state : batch) legal.push_back(GenerateDoubleMoveLegalMoves(state));
      const auto decisions = evaluate(batch, legal);
      if (decisions.size() != batch.size()) throw Exception("Decision batch size mismatch");
      for (size_t i = 0; i < batch.size(); ++i) {
        const auto& decision = decisions[i];
        if (decision.priors.size() != legal[i].size() ||
            decision.selected >= legal[i].size()) throw Exception("Invalid Black decision");
        const auto best = decision.selected;
        const Move move = legal[i][best];
        float runner_up = 0;
        for (size_t j = 0; j < decision.priors.size(); ++j) {
          if (j != best) runner_up = std::max(runner_up, decision.priors[j]);
        }
        auto absolute_move = move;
        absolute_move.Flip();
        trace << depth << '\t' << PolicyMateUci(batch[i]) << '\t'
              << absolute_move.ToString(false) << '\t' << std::setprecision(9)
              << decision.priors[best] << '\t' << runner_up << '\t'
              << move.is_ability_activation() << '\t' << decision.visits << '\n';
        if (move.is_ability_activation()) { ++depth_pruned; continue; }
        auto next = WithMoves(batch[i], {move});
        if (next.GetPositionHistory().ComputeGameResult() != GameResult::UNDECIDED) {
          ++terminal;
        } else {
          frontier.push_back(std::move(next));
        }
      }
      queries += batch.size();
      depth_queries += batch.size();
    }
    pruned += depth_pruned;
    trace.flush();
    output << "{\"type\":\"black_replies\",\"white_turn\":" << depth
           << ",\"queries\":" << depth_queries << ",\"pruned_ability\":" << depth_pruned
           << ",\"terminal\":" << terminal << ",\"next_frontier\":" << frontier.size()
           << ",\"phase_seconds\":" << Elapsed(inference_started) << "}" << std::endl;
    if (status == "time_limit") break;
    if (frontier.empty()) { status = "restricted_tree_exhausted"; break; }
  }
  output << "{\"type\":\"summary\",\"status\":\"" << status
         << "\",\"complete_white_turns\":" << complete_depth
         << ",\"policy_queries\":" << queries << ",\"double_pairs\":" << pairs
         << ",\"pruned_black_activation\":" << pruned
         << ",\"elapsed_seconds\":" << Elapsed(started) << "}" << std::endl;
}
}  // namespace lczero
