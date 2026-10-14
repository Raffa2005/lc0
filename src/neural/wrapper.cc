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

#include "neural/wrapper.h"

#include <algorithm>
#include <numeric>

#include "neural/encoder.h"
#include "neural/shared_params.h"
#include "utils/atomic_vector.h"
#include "utils/fastmath.h"
#include "utils/trace.h"

namespace lczero {
namespace {

FillEmptyHistory EncodeHistoryFill(std::string history_fill) {
  if (history_fill == "fen_only") return FillEmptyHistory::FEN_ONLY;
  if (history_fill == "always") return FillEmptyHistory::ALWAYS;
  if (history_fill == "repeat_current") {
    return FillEmptyHistory::REPEAT_CURRENT;
  }
  assert(history_fill == "no");
  return FillEmptyHistory::NO;
}

class NetworkAsBackend : public Backend {
 public:
  NetworkAsBackend(std::unique_ptr<Network> network, const OptionsDict& options)
      : network_(std::move(network)),
        backend_opts_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
        weights_path_(
            options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    UpdateConfiguration(options);
    const NetworkCapabilities& caps = network_->GetCapabilities();
    attrs_.has_mlh = caps.has_mlh();
    attrs_.has_wdl = caps.has_wdl();
    attrs_.runs_on_cpu = network_->IsCpu();
    attrs_.suggested_num_search_threads = network_->GetThreads();
    attrs_.recommended_batch_size = network_->GetMiniBatchSize();
    attrs_.maximum_batch_size = 1024;
    input_format_ = caps.input_format;
  }

  BackendAttributes GetAttributes() const override { return attrs_; }
  std::unique_ptr<BackendComputation> CreateComputation() override;
  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    if (backend_opts_ !=
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId)) {
      return NEED_RESTART;
    }
    if (weights_path_ !=
        options.Get<std::string>(SharedBackendParams::kWeightsId)) {
      return NEED_RESTART;
    }
    softmax_policy_temperature_ =
        1.0f / options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    fill_empty_history_ = EncodeHistoryFill(
        options.Get<std::string>(SharedBackendParams::kHistoryFill));
    return UPDATE_OK;
  }

 private:
  std::unique_ptr<Network> network_;
  BackendAttributes attrs_;
  pblczero::NetworkFormat::InputFormat input_format_;
  float softmax_policy_temperature_;
  FillEmptyHistory fill_empty_history_;
  const std::string backend_opts_;
  const std::string weights_path_;

  friend class NetworkAsBackendComputation;
};

class NetworkAsBackendComputation : public BackendComputation {
 public:
  NetworkAsBackendComputation(NetworkAsBackend* backend)
      : backend_(backend),
        computation_(backend_->network_->NewComputation()),
        entries_(backend_->attrs_.maximum_batch_size) {}

  size_t UsedBatchSize() const override { return entries_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    int transform;
    const size_t idx = entries_.emplace_back(Entry{
        .input = EncodePositionForNN(backend_->input_format_, pos.pos, 8,
                                     backend_->fill_empty_history_, &transform),
        .legal_moves = MoveList(pos.legal_moves.begin(), pos.legal_moves.end()),
        .result = result,
        .transform = 0});
    entries_[idx].transform = transform;
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    for (auto& entry : entries_) computation_->AddInput(std::move(entry.input));
    computation_->ComputeBlocking();
    LCTRACE_FUNCTION_SCOPE;
    for (size_t i = 0; i < entries_.size(); ++i) {
      const EvalResultPtr& result = entries_[i].result;
      if (result.q) *result.q = computation_->GetQVal(i);
      if (result.d) *result.d = computation_->GetDVal(i);
      if (result.m) *result.m = computation_->GetMVal(i);
      if (!result.p.empty()) SoftmaxPolicy(result.p, computation_.get(), i);
    }
  }

  void SoftmaxPolicy(std::span<float> dst,
                     const NetworkComputation* computation, int idx) {
    LCTRACE_FUNCTION_SCOPE;
    const std::vector<Move>& moves = entries_[idx].legal_moves;
    const int transform = entries_[idx].transform;

    // Double-move split-policy heads with a burn gate return log-probabilities
    // already composed over the full 3716-action surface:
    //   log((1-gate) * softmax(normal_logits over all 1858 actions))
    //   log(gate       * softmax(ability_logits over all 1858 actions))
    // Training, however, uses illegal_moves=MASK and normalizes each branch on
    // legal moves only. If search then softmaxes the full-composed values over
    // legal moves, illegal ability logits can absorb almost all ability-branch
    // mass and make the visible burn prior much smaller than the trained gate.
    //
    // Detect this composed-logprob representation by summing exp(policy) over
    // the full surface. Raw-logit heads do not sum to ~1 and fall back to the
    // ordinary legal softmax below. For composed heads, recover the raw gate as
    // total ability mass, then renormalize normal and ability branches
    // separately over the legal moves before applying that gate.
    float full_normal = 0.0f;
    float full_ability = 0.0f;
    for (int j = 0; j < 1858; ++j) {
      full_normal += FastExp(computation->GetPVal(idx, j));
      full_ability += FastExp(computation->GetPVal(idx, 1858 + j));
    }
    const float full_total = full_normal + full_ability;
    const bool looks_like_composed_split_policy =
        full_total > 0.5f && full_total < 1.5f && full_normal > 1e-8f;

    if (looks_like_composed_split_policy) {
      std::vector<float> legal_logits(moves.size());
      float normal_max = -std::numeric_limits<float>::infinity();
      float ability_max = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < moves.size(); ++i) {
        const Move& move = moves[i];
        const int nn_idx =
            move.is_ability_activation()
                ? MoveToNNIndex(move.WithoutAbilityActivation(), transform) +
                      1858
                : MoveToNNIndex(move, transform);
        const float logp = computation->GetPVal(idx, nn_idx);
        legal_logits[i] = logp;
        if (move.is_ability_activation()) {
          ability_max = std::max(ability_max, logp);
        } else {
          normal_max = std::max(normal_max, logp);
        }
      }

      const float temperature = backend_->softmax_policy_temperature_;
      std::vector<float> legal_probs(moves.size());
      float legal_normal = 0.0f;
      float legal_ability = 0.0f;
      for (size_t i = 0; i < moves.size(); ++i) {
        const bool is_ability = moves[i].is_ability_activation();
        const float branch_max = is_ability ? ability_max : normal_max;
        const float prob = FastExp((legal_logits[i] - branch_max) * temperature);
        legal_probs[i] = prob;
        if (is_ability) {
          legal_ability += prob;
        } else {
          legal_normal += prob;
        }
      }

      const bool has_legal_normal = legal_normal > 0.0f;
      const bool has_legal_ability = legal_ability > 0.0f;
      const float normal_gate = full_normal / full_total;
      const float ability_gate = full_ability / full_total;
      const float legal_gate_total =
          (has_legal_normal ? normal_gate : 0.0f) +
          (has_legal_ability ? ability_gate : 0.0f);
      const float legal_gate_scale =
          legal_gate_total > 0.0f ? 1.0f / legal_gate_total : 1.0f;

      for (size_t i = 0; i < moves.size(); ++i) {
        if (moves[i].is_ability_activation()) {
          dst[i] = has_legal_ability
                       ? legal_gate_scale * ability_gate *
                             (legal_probs[i] / legal_ability)
                       : 0.0f;
        } else {
          dst[i] = has_legal_normal
                       ? legal_gate_scale * normal_gate *
                             (legal_probs[i] / legal_normal)
                       : 0.0f;
        }
      }
      return;
    }

    // Copy the values to the destination array and compute the maximum.
    const float max_p = std::accumulate(
        moves.begin(), moves.end(), -std::numeric_limits<float>::infinity(),
        [&, counter = 0](float max_p, const Move& move) mutable {
          // Double-move variant: ability-activation moves map to the second
          // half of the 3716-output policy head (indices 1858..3715).
          const uint16_t nn_idx =
              move.is_ability_activation()
                  ? MoveToNNIndex(move.WithoutAbilityActivation(), transform) +
                        1858
                  : MoveToNNIndex(move, transform);
          return std::max(max_p, dst[counter++] =
                                     computation->GetPVal(idx, nn_idx));
        });
    // Compute the softmax and compute the total.
    const float temperature = backend_->softmax_policy_temperature_;
    float total = std::accumulate(
        dst.begin(), dst.end(), 0.0f, [&](float total, float& val) {
          return total + (val = FastExp((val - max_p) * temperature));
        });
    const float scale = total > 0.0f ? 1.0f / total : 1.0f;
    // Scale the values to sum to 1.0.
    std::for_each(dst.begin(), dst.end(), [&](float& val) { val *= scale; });
  }

 private:
  struct Entry {
    InputPlanes input;
    MoveList legal_moves;
    EvalResultPtr result;
    int transform;
  };

  NetworkAsBackend* backend_;
  std::unique_ptr<NetworkComputation> computation_;
  AtomicVector<Entry> entries_;
};

std::unique_ptr<BackendComputation> NetworkAsBackend::CreateComputation() {
  return std::make_unique<NetworkAsBackendComputation>(this);
}

}  // namespace

NetworkAsBackendFactory::NetworkAsBackendFactory(const std::string& name,
                                                 FactoryFunc factory,
                                                 int priority)
    : name_(name), factory_(factory), priority_(priority) {}

std::unique_ptr<Backend> NetworkAsBackendFactory::Create(
    const OptionsDict& options) {
  const std::string backend_options =
      options.Get<std::string>(SharedBackendParams::kBackendOptionsId);
  OptionsDict network_options;
  network_options.AddSubdictFromString(backend_options);

  std::string net_path =
      options.Get<std::string>(SharedBackendParams::kWeightsId);
  std::optional<WeightsFile> weights = LoadWeights(net_path);
  std::unique_ptr<Network> network =
      factory_(std::move(weights), network_options);
  network_options.CheckAllOptionsRead(name_);
  return std::make_unique<NetworkAsBackend>(std::move(network), options);
}

}  // namespace lczero
