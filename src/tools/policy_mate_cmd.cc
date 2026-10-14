// SPDX-License-Identifier: GPL-3.0-or-later
#include "tools/policy_mate.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>

#include "neural/dual_backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/stoppers.h"
#include "utils/optionsparser.h"

namespace lczero {
void PolicyMateCommand() {
  const OptionId turns{"white-turns", "", "Maximum White turn to exhaust."};
  const OptionId threads{"threads", "", "CPU workers for legal mate enumeration."};
  const OptionId batch{"batch-size", "", "Black policy inference batch size."};
  const OptionId seconds{"seconds", "", "Search wall clock budget, excluding network loading."};
  const OptionId frontier{"max-frontier", "", "Maximum pending Black positions."};
  const OptionId trace{"trace", "", "Output TSV of every Black policy decision."};
  const OptionId prefix_file{"prefix-file", "", "Evaluate only: one startpos UCI prefix per line."};
  const OptionId verify_file{"verify-mates", "", "Rules-only verification: one complete UCI mate line per row."};
  const OptionId visits{"black-visits", "", "Black MCTS visits; zero selects raw top policy."};
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  classic::SearchParams::Populate(&options);
  options.Add<IntOption>(turns, 1, 8) = 4;
  options.Add<IntOption>(threads, 1, 128) = 24;
  options.Add<IntOption>(batch, 1, 1024) = 256;
  options.Add<IntOption>(seconds, 1, 3600) = 600;
  options.Add<IntOption>(frontier, 1, 2000000) = 1000000;
  options.Add<StringOption>(trace) = "policy_mate_trace.tsv";
  options.Add<StringOption>(prefix_file) = "";
  options.Add<StringOption>(verify_file) = "";
  options.Add<IntOption>(visits, 0, 1000000) = 0;
  if (!options.ProcessAllFlags()) return;
  const auto& dict = options.GetOptionsDict();
  if (!dict.Get<std::string>(verify_file).empty()) {
    std::ifstream input(dict.Get<std::string>(verify_file));
    if (!input) throw Exception("Cannot read mate verification file");
    std::string line;
    while (std::getline(input, line)) {
      std::istringstream stream(line);
      std::vector<std::string> moves;
      std::string move;
      while (stream >> move) moves.push_back(move);
      const auto state = MakeGameStateFromUci(ChessBoard::kStartposFen, moves);
      if (state.GetPositionHistory().ComputeGameResult() != GameResult::WHITE_WON) {
        throw Exception("Not White checkmate: " + line);
      }
      std::cout << "{\"type\":\"verified_mate\",\"moves_uci\":\""
                << PolicyMateUci(state) << "\"}" << std::endl;
    }
    return;
  }
  OptionsDict backend_options;
  backend_options.AddSubdictFromString(dict.Get<std::string>(SharedBackendParams::kBackendOptionsId));
  const int black_visits = dict.Get<int>(visits);
  if (!black_visits && dict.Get<std::string>(SharedBackendParams::kBackendId).starts_with("cuda") &&
      dict.Get<int>(batch) > backend_options.GetOrDefault<int>("max_batch", 1024)) {
    throw Exception("batch-size exceeds CUDA max_batch");
  }
  auto backend = BackendManager::Get()->CreateFromParams(dict);
  const auto fallback = dict.Get<std::string>(SharedBackendParams::kFallbackWeightsId);
  if (!fallback.empty()) {
    OptionsDict fallback_options(&dict);
    fallback_options.Set<std::string>(SharedBackendParams::kWeightsId, fallback);
    backend = std::make_unique<DualBackend>(std::move(backend),
        BackendManager::Get()->CreateFromParams(fallback_options));
  }
  if (!black_visits && dict.Get<int>(batch) > backend->GetAttributes().maximum_batch_size) {
    throw Exception("batch-size exceeds backend maximum");
  }
  PolicyMateEvaluator evaluate = [&](std::span<const GameState> states,
                                     const std::vector<MoveList>& legal) {
    std::vector<std::vector<Position>> histories;
    histories.reserve(states.size());
    for (const auto& state : states) histories.push_back(state.GetPositions());
    std::vector<EvalPosition> input;
    for (size_t i = 0; i < states.size(); ++i) input.push_back({histories[i], legal[i]});
    auto result = backend->EvaluateBatch(input);
    std::vector<PolicyMateDecision> decisions;
    for (size_t i = 0; i < result.size(); ++i) {
      const auto best = PolicyMateBest(result[i].p, legal[i]);
      decisions.push_back({std::move(result[i].p), best});
    }
    return decisions;
  };
  if (black_visits) {
    evaluate = [&](std::span<const GameState> states,
                   const std::vector<MoveList>& legal) {
      std::vector<PolicyMateDecision> decisions(states.size());
      std::atomic<size_t> cursor{0};
      std::exception_ptr error;
      std::mutex error_mutex;
      auto worker = [&] {
        try {
          for (size_t i = cursor.fetch_add(1); i < states.size(); i = cursor.fetch_add(1)) {
            // Each independent reply starts with an empty tree and full history.
            classic::NodeTree tree;
            tree.ResetToPosition(states[i]);
            classic::Search search(tree, backend.get(),
                std::make_unique<CallbackUciResponder>(
                    [](const BestMoveInfo&) {}, [](const std::vector<ThinkingInfo>&) {}),
                MoveList(), std::chrono::steady_clock::now(),
                std::make_unique<classic::VisitsStopper>(black_visits, false),
                false, false, dict, nullptr);
            search.RunBlocking(1);
            auto move = search.GetBestMove().first;
            if (states[i].CurrentPosition().IsBlackToMove()) move.Flip();
            auto& decision = decisions[i];
            decision.selected = std::find(legal[i].begin(), legal[i].end(), move) - legal[i].begin();
            if (decision.selected == legal[i].size()) throw Exception("Search returned illegal move");
            decision.visits = search.GetTotalPlayouts();
            decision.priors.resize(legal[i].size());
            for (const auto& edge : tree.GetCurrentHead()->Edges()) {
              const auto index = std::find(legal[i].begin(), legal[i].end(), edge.GetMove()) - legal[i].begin();
              if (index == static_cast<int64_t>(legal[i].size())) throw Exception("Unexpected root edge");
              decision.priors[index] = edge.GetP();
            }
          }
        } catch (...) {
          std::lock_guard lock(error_mutex);
          if (!error) error = std::current_exception();
        }
      };
      std::vector<std::thread> workers;
      const auto count = std::min<size_t>(dict.Get<int>(threads), states.size());
      for (size_t t = 1; t < count; ++t) workers.emplace_back(worker);
      worker();
      for (auto& thread : workers) thread.join();
      if (error) std::rethrow_exception(error);
      return decisions;
    };
  }
  const auto filename = dict.Get<std::string>(prefix_file);
  if (!filename.empty()) {
    std::ifstream input(filename);
    if (!input) throw Exception("Cannot read prefix file: " + filename);
    std::vector<GameState> states;
    std::vector<MoveList> legal;
    auto flush = [&] {
      if (states.empty()) return;
      const auto decisions = evaluate(states, legal);
      for (size_t i = 0; i < states.size(); ++i) {
        auto move = legal[i][decisions[i].selected];
        if (states[i].CurrentPosition().IsBlackToMove()) move.Flip();
        std::cout << "{\"type\":\"policy\",\"moves_uci\":\"" << PolicyMateUci(states[i])
                  << "\",\"bestmove\":\"" << move.ToString(false)
                  << "\",\"visits\":" << decisions[i].visits << "}" << std::endl;
      }
      states.clear();
      legal.clear();
    };
    std::string line;
    while (std::getline(input, line)) {
      std::istringstream stream(line);
      std::vector<std::string> moves;
      std::string move;
      while (stream >> move) moves.push_back(move);
      auto state = MakeGameStateFromUci(ChessBoard::kStartposFen, moves);
      auto actions = GenerateDoubleMoveLegalMoves(state);
      if (actions.empty()) throw Exception("Terminal validation prefix");
      states.push_back(std::move(state));
      legal.push_back(std::move(actions));
      if (states.size() == static_cast<size_t>(dict.Get<int>(batch))) flush();
    }
    flush();
    return;
  }
  std::ofstream trace_output(dict.Get<std::string>(trace));
  if (!trace_output) throw Exception("Cannot write policy trace");
  trace_output.exceptions(std::ios::badbit | std::ios::failbit);
  SearchPolicyMates(MakeGameStateFromUci(ChessBoard::kStartposFen, {}),
      {.white_turns = dict.Get<int>(turns), .threads = dict.Get<int>(threads),
       .batch_size = dict.Get<int>(batch), .seconds = dict.Get<int>(seconds),
       .max_frontier = static_cast<size_t>(dict.Get<int>(frontier)),
       .black_visits = black_visits},
      evaluate, std::cout, trace_output);
}
}  // namespace lczero
