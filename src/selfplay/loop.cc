/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "selfplay/loop.h"

#include <optional>

#include "selfplay/tournament.h"
#include "utils/configfile.h"
#include "utils/optionsparser.h"

namespace lczero {

namespace {
const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console."};
}  // namespace

SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}

void SelfPlayLoop::Run() {
  SelfPlayTournament::PopulateOptions(&options_);

  options_.Add<StringOption>(kLogFileId);

  if (!options_.ProcessAllFlags()) return;

  Logging::Get().SetFilename(
      options_.GetOptionsDict().Get<std::string>(kLogFileId));

  // Send id before starting tournament to allow wrapping client to know
  // who we are.
  uci_responder_->SendId();
  SelfPlayTournament tournament(
      options_.GetOptionsDict(), uci_responder_,
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  tournament.RunBlocking();
}

void SelfPlayLoop::SendGameInfo(const GameInfo& info) {
  std::vector<std::string> responses;
  // Send separate resign report before gameready as client gameready parsing
  // will easily get confused by adding new parameters as both training file
  // and move list potentially contain spaces.
  if (info.min_false_positive_threshold) {
    std::string resign_res = "resign_report";
    resign_res +=
        " fp_threshold " + std::to_string(*info.min_false_positive_threshold);
    responses.push_back(resign_res);
  }
  std::string res = "gameready";
  if (!info.training_filename.empty())
    res += " trainingfile " + info.training_filename;
  if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
  if (info.opening_id) {
    res += " openingid " + *info.opening_id;
    res += std::string(" openingterminal ") +
           (info.opening_terminal ? "true" : "false");
  }
  res += " play_start_ply " + std::to_string(info.play_start_ply);
  if (info.is_black)
    res += " player1 " + std::string(*info.is_black ? "black" : "white");
  if (info.game_result != GameResult::UNDECIDED) {
    if (info.soft_adjudicated) {
      res += " result adjudicated q " + std::to_string(info.soft_q) +
             " d " + std::to_string(info.soft_d);
    } else {
      res += std::string(" result ") +
             ((info.game_result == GameResult::DRAW)        ? "draw"
              : (info.game_result == GameResult::WHITE_WON) ? "whitewon"
                                                            : "blackwon");
    }
  }
  if (!info.moves.empty()) {
    res += " moves";
    for (const auto& move : info.moves) res += " " + move.ToString(true);
  }
  if (!info.initial_fen.empty() &&
      info.initial_fen != ChessBoard::kStartposFen) {
    res += " from_fen " + info.initial_fen;
  }
  responses.push_back(res);
  uci_responder_->SendRawResponses(responses);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  const int total_games = info.game_count[0] + info.game_count[1];
  const float total_score = info.score_sum[0] + info.score_sum[1];

  float percentage = -1;
  std::optional<float> elo;

  // Calculate percentage if any games at all (avoid divide by 0).
  if (total_games > 0) {
    percentage = total_score / total_games;
  }
  // Calculate Elo if percentage strictly between 0 and 1.
  if (percentage > 0 && percentage < 1) {
    elo = -400 * log(1 / percentage - 1) / log(10);
  }

  std::ostringstream oss;
  oss << "tournamentstatus";
  if (info.finished) oss << " final";
  oss << " P1: " << total_games << " games";
  oss << " score " << std::fixed << std::setprecision(3) << total_score
      << "/" << total_games;

  if (percentage > 0) {
    oss << " Win: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (percentage * 100.0f) << "%";
  }
  if (elo) {
    oss << " Elo: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*elo);
  }

  auto avg = [](float sum, int count) {
    return count > 0 ? sum / count : 0.0f;
  };
  oss << " P1-W: " << info.game_count[0] << " games"
      << " avg " << std::fixed << std::setprecision(3)
      << avg(info.score_sum[0], info.game_count[0]);
  oss << " P1-B: " << info.game_count[1] << " games"
      << " avg " << std::fixed << std::setprecision(3)
      << avg(info.score_sum[1], info.game_count[1]);

  const double npm =
      info.move_count_ > 0
          ? static_cast<double>(info.nodes_total_) / info.move_count_
          : 0.0;
  oss << " npm " + std::to_string(npm);
  oss << " nodes " + std::to_string(info.nodes_total_);
  oss << " moves " + std::to_string(info.move_count_);
  uci_responder_->SendRawResponse(oss.str());
}

}  // namespace lczero
