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

#include "chess/position.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>

#include "chess/types.h"
#include "utils/exception.h"

namespace lczero {

Position::Position(const Position& parent, Move m)
    : rule50_ply_(parent.rule50_ply_ + 1), ply_count_(parent.ply_count_ + 1) {
  us_board_ = parent.us_board_;
  const bool is_zeroing = us_board_.ApplyMove(m);
  us_board_.Mirror();
  if (is_zeroing) rule50_ply_ = 0;
  // After Mirror(), perspectives swap: our <-> their.
  our_doublemove_available_ = parent.their_doublemove_available_;
  their_doublemove_available_ = parent.our_doublemove_available_;
  is_mid_doublemove_ = false;
}

Position::Position(const Position& parent, Move m, bool is_first_of_double)
    : rule50_ply_(parent.rule50_ply_ + 1), ply_count_(parent.ply_count_ + 1) {
  us_board_ = parent.us_board_;
  const bool is_zeroing = us_board_.ApplyMove(m);
  if (!is_first_of_double) {
    us_board_.Mirror();
  }
  if (is_zeroing) rule50_ply_ = 0;

  if (is_first_of_double) {
    // Same player moves again; ability is consumed, no perspective swap.
    our_doublemove_available_ = false;
    their_doublemove_available_ = parent.their_doublemove_available_;
    is_mid_doublemove_ = true;
  } else {
    // Normal move: perspectives swap after Mirror().
    our_doublemove_available_ = parent.their_doublemove_available_;
    their_doublemove_available_ = parent.our_doublemove_available_;
    is_mid_doublemove_ = false;
  }
}

Position::Position(const ChessBoard& board, int rule50_ply, int game_ply)
    : rule50_ply_(rule50_ply), repetitions_(0), ply_count_(game_ply) {
  us_board_ = board;
}

// Parse an optional 7th FEN field that encodes double-move ability state.
// Format: one token with characters drawn from {W, B, w, b, m, -}.
//   'W' or 'w' -> White still has the double-move ability.
//   'B' or 'b' -> Black still has the double-move ability.
//   'm'        -> currently in a mid-doublemove (between the two halves).
//   '-'        -> placeholder, ignored.
// If the 7th field is absent we keep the Position defaults
// (both abilities available, not mid-doublemove). Examples:
//   "WB"     both have ability (default)
//   "W"      only white has ability
//   "-B"     only black has ability
//   "Wm"    white has ability AND mid-doublemove (rare; legal positions only)
static void ParseAbilityField(std::string_view tok, bool* white_has,
                              bool* black_has, bool* mid_double) {
  *white_has = false;
  *black_has = false;
  *mid_double = false;
  for (char c : tok) {
    switch (c) {
      case 'W': case 'w': *white_has = true; break;
      case 'B': case 'b': *black_has = true; break;
      case 'M': case 'm': *mid_double = true; break;
      case '-': break;
      default:
        throw Exception(
            std::string("Invalid character '") + c +
            "' in ability field; expected W/B/m/-: " + std::string(tok));
    }
  }
}

Position Position::FromFen(std::string_view fen) {
  Position pos;
  // Split off any 7th-or-later whitespace-delimited token; that is our
  // double-move ability extension. Standard FEN has 6 fields; pass only
  // those (plus any trailing whitespace) to SetFromFen, which is strict.
  std::string_view base_fen = fen;
  std::string_view ability_tok;
  {
    size_t pos_idx = 0;
    int token_count = 0;
    bool in_token = false;
    size_t end_of_6 = fen.size();
    for (; pos_idx < fen.size(); ++pos_idx) {
      bool ws = fen[pos_idx] == ' ' || fen[pos_idx] == '\t';
      if (!ws && !in_token) {
        in_token = true;
        ++token_count;
        if (token_count == 7) {
          // Step back so end_of_6 points to the start of the 7th token's
          // leading whitespace, keeping a clean base_fen for SetFromFen.
          while (end_of_6 > 0 &&
                 (fen[end_of_6 - 1] == ' ' || fen[end_of_6 - 1] == '\t')) {
            --end_of_6;
          }
          break;
        }
      } else if (ws && in_token) {
        in_token = false;
        end_of_6 = pos_idx;
      }
    }
    if (token_count >= 7) {
      base_fen = fen.substr(0, end_of_6);
      // Find the end of the 7th token (next whitespace or end of input).
      size_t start = pos_idx;
      size_t stop = start;
      while (stop < fen.size() && fen[stop] != ' ' && fen[stop] != '\t') {
        ++stop;
      }
      ability_tok = fen.substr(start, stop - start);
    }
  }

  pos.us_board_.SetFromFen(std::string(base_fen), &pos.rule50_ply_,
                           &pos.ply_count_);

  if (!ability_tok.empty()) {
    bool white_has, black_has, mid_double;
    ParseAbilityField(ability_tok, &white_has, &black_has, &mid_double);
    // Translate global-color flags to perspective-relative fields. After
    // SetFromFen the board is mirrored iff black is to move.
    const bool black_to_move = pos.us_board_.flipped();
    pos.our_doublemove_available_ = black_to_move ? black_has : white_has;
    pos.their_doublemove_available_ = black_to_move ? white_has : black_has;
    pos.is_mid_doublemove_ = mid_double;
  }

  return pos;
}

uint64_t Position::Hash() const {
  return HashCat({us_board_.Hash(), static_cast<unsigned long>(repetitions_),
                  static_cast<unsigned long>(our_doublemove_available_),
                  static_cast<unsigned long>(their_doublemove_available_),
                  static_cast<unsigned long>(is_mid_doublemove_)});
}

std::string Position::DebugString() const {
  std::string fen = PositionToFen(*this);
  std::replace(fen.begin(), fen.end(), ' ', '_');
  return "https://lc0.org/fen/" + fen;
}

GameResult operator-(const GameResult& res) {
  return res == GameResult::BLACK_WON   ? GameResult::WHITE_WON
         : res == GameResult::WHITE_WON ? GameResult::BLACK_WON
                                        : res;
}

GameResult PositionHistory::ComputeGameResult() const {
  const auto& last = Last();
  const auto& board = last.GetBoard();

  // During a double-move (between move 1 and move 2), the game is never over.
  // The current player still needs to make move 2. Checkmate, stalemate, and
  // draw conditions are only evaluated at the end of complete turns.
  if (last.IsMidDoublemove()) {
    return GameResult::UNDECIDED;
  }

  auto legal_moves = board.GenerateLegalMoves();
  if (legal_moves.empty()) {
    if (board.IsUnderCheck()) {
      // Checkmate.
      return last.IsBlackToMove() ? GameResult::WHITE_WON
                                  : GameResult::BLACK_WON;
    }
    // Stalemate.
    return GameResult::DRAW;
  }

  if (!board.HasMatingMaterial()) return GameResult::DRAW;
  if (last.GetRule50Ply() >= 100) return GameResult::DRAW;
  if (last.GetRepetitions() >= 2) return GameResult::DRAW;

  return GameResult::UNDECIDED;
}

void PositionHistory::Reset(const ChessBoard& board, int rule50_ply,
                            int game_ply) {
  positions_.clear();
  positions_.emplace_back(board, rule50_ply, game_ply);
}

void PositionHistory::Reset(const Position& pos) {
  positions_.clear();
  positions_.push_back(pos);
}

void PositionHistory::Append(Move m) {
  // TODO(mooskagh) That should be emplace_back(Last(), m), but MSVS STL
  //                has a bug in implementation of emplace_back, when
  //                reallocation happens. (it also reallocates Last())
  positions_.push_back(Position(Last(), m));
  int cycle_length;
  int repetitions = ComputeLastMoveRepetitions(&cycle_length);
  positions_.back().SetRepetitions(repetitions, cycle_length);
}

void PositionHistory::Append(Move m, bool is_first_of_double) {
  positions_.push_back(Position(Last(), m, is_first_of_double));
  if (!is_first_of_double) {
    // Only count repetitions for positions that end a full turn.
    int cycle_length;
    int repetitions = ComputeLastMoveRepetitions(&cycle_length);
    positions_.back().SetRepetitions(repetitions, cycle_length);
  }
  // Mid-double-move positions do not count for repetition.
}

int PositionHistory::ComputeLastMoveRepetitions(int* cycle_length) const {
  *cycle_length = 0;
  const auto& last = positions_.back();
  // TODO(crem) implement hash/cache based solution.
  if (last.GetRule50Ply() < 4) return 0;

  for (int idx = positions_.size() - 5; idx >= 0; idx -= 2) {
    const auto& pos = positions_[idx];
    if (pos.GetBoard() == last.GetBoard()) {
      *cycle_length = positions_.size() - 1 - idx;
      return 1 + pos.GetRepetitions();
    }
    if (pos.GetRule50Ply() < 2) return 0;
  }
  return 0;
}

bool PositionHistory::DidRepeatSinceLastZeroingMove() const {
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (iter->GetRepetitions() > 0) return true;
    if (iter->GetRule50Ply() == 0) return false;
  }
  return false;
}

uint64_t PositionHistory::HashLast(int positions) const {
  uint64_t hash = positions;
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (!positions--) break;
    hash = HashCat(hash, iter->Hash());
  }
  return HashCat(hash, Last().GetRule50Ply());
}

std::string PositionToFen(const Position& pos) {
  std::string result = BoardToFen(pos.GetBoard());
  result += " " + std::to_string(pos.GetRule50Ply());
  result += " " + std::to_string(
                      (pos.GetGamePly() + (pos.IsBlackToMove() ? 1 : 2)) / 2);
  // Append double-move ability state only when it differs from the default
  // (both have ability, not mid-doublemove). This keeps standard FENs clean.
  const bool black_to_move = pos.IsBlackToMove();
  const bool white_has = black_to_move ? pos.TheirDoublemoveAvailable()
                                       : pos.OurDoublemoveAvailable();
  const bool black_has = black_to_move ? pos.OurDoublemoveAvailable()
                                       : pos.TheirDoublemoveAvailable();
  const bool mid = pos.IsMidDoublemove();
  if (!white_has || !black_has || mid) {
    std::string tok;
    tok += white_has ? 'W' : '-';
    tok += black_has ? 'B' : '-';
    if (mid) tok += 'm';
    result += " " + tok;
  }
  return result;
}
}  // namespace lczero
