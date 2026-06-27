/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include "chess/gamestate.h"

#include <algorithm>
#include <numeric>

namespace lczero {

GameState MakeGameStateFromUci(const std::string& fen,
                               const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(fen);
  ChessBoard cur_board = state.startpos.GetBoard();
  state.moves.reserve(moves.size());
  for (const auto& move : moves) {
    Move m = cur_board.ParseMove(move);
    state.moves.push_back(m);
    cur_board.ApplyMove(m);
    if (!m.is_ability_activation()) {
      cur_board.Mirror();
    }
  }
  return state;
}

Position GameState::CurrentPosition() const {
  return std::accumulate(
      moves.begin(), moves.end(), startpos,
      [](const Position& pos, Move m) {
        if (m.is_ability_activation()) {
          return Position(pos, m.WithoutAbilityActivation(),
                          /*is_first_of_double=*/true);
        }
        return Position(pos, m);
      });
}

std::vector<Position> GameState::GetPositions() const {
  std::vector<Position> positions;
  positions.reserve(moves.size() + 1);
  positions.push_back(startpos);
  std::transform(moves.begin(), moves.end(), std::back_inserter(positions),
                 [&](Move m) {
                   if (m.is_ability_activation()) {
                     return Position(positions.back(),
                                     m.WithoutAbilityActivation(),
                                     /*is_first_of_double=*/true);
                   }
                   return Position(positions.back(), m);
                 });
  return positions;
}

}  // namespace lczero
