#include "cloze/sample.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

namespace cloze {

std::vector<Candidate> sample_candidates(const ForwardResult& fwd,
                                         const std::vector<int>& positions,
                                         const SampleOpts& opts) {
    if (static_cast<int>(positions.size()) != fwd.n_requested) {
        throw std::invalid_argument("positions count != logits rows");
    }
    if (opts.temperature > 0.0 && opts.rng == nullptr) {
        throw std::invalid_argument("temperature > 0 requires an rng");
    }
    const int vocab = fwd.vocab;

    // Tokens to penalize: everything already on the board except the mask / out-of-range ids.
    std::set<int> penalize;
    if (opts.rep_penalty != 1.0 && opts.board != nullptr) {
        for (int id : *opts.board)
            if (id != opts.mask_token && id >= 0 && id < vocab) penalize.insert(id);
    }
    const bool greedy = !(opts.temperature > 0.0);

    std::vector<Candidate> out;
    out.reserve(positions.size());
    std::vector<double> x(static_cast<size_t>(vocab));

    for (int r = 0; r < fwd.n_requested; ++r) {
        const float* row = fwd.row(r);
        for (int t = 0; t < vocab; ++t) x[t] = static_cast<double>(row[t]);

        // Repetition penalty (CTRL/HF convention): pull already-seen tokens' logits toward 0.
        for (int t : penalize) x[t] = x[t] > 0.0 ? x[t] / opts.rep_penalty : x[t] * opts.rep_penalty;
        // Temperature: flatten (T>1) / sharpen (T<1) before the softmax.
        if (!greedy)
            for (int t = 0; t < vocab; ++t) x[t] /= opts.temperature;

        // Stable float64 softmax (subtract the max). On the default path (no penalty, greedy) this is
        // bit-for-bit the prior greedy confidence: argmax prob == 1 / sum exp(logit - max).
        double mx = x[0];
        for (int t = 1; t < vocab; ++t)
            if (x[t] > mx) mx = x[t];
        double denom = 0.0;
        for (int t = 0; t < vocab; ++t) { x[t] = std::exp(x[t] - mx); denom += x[t]; }
        for (int t = 0; t < vocab; ++t) x[t] /= denom;  // x is now a probability vector

        int token;
        double conf;
        if (greedy) {
            token = 0;
            double best = x[0];  // argmax; ties resolve to the lower id (== numpy.argmax)
            for (int t = 1; t < vocab; ++t)
                if (x[t] > best) { best = x[t]; token = t; }
            conf = x[token];
        } else {
            // Optional Ollama-style truncation on the softmax distribution: top-k, then top-p (nucleus),
            // renormalize, then the inverse-CDF draw. Off by default (top_k<=0 and top_p>=1 keep the full
            // distribution), and it touches ONLY this stochastic branch -- greedy above is bit-identical,
            // so the receipts-critical goldens are unaffected. The draw stays seeded/deterministic.
            const bool trunc = (opts.top_k > 0 && opts.top_k < vocab) || (opts.top_p > 0.0 && opts.top_p < 1.0);
            std::vector<double> orig;   // true softmax probs, kept for the honest committed-token confidence
            if (trunc) {
                orig = x;
                std::vector<int> idx(static_cast<size_t>(vocab));
                for (int t = 0; t < vocab; ++t) idx[t] = t;
                const int kk = (opts.top_k > 0 && opts.top_k < vocab) ? opts.top_k : vocab;
                std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                                  [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
                const double p = (opts.top_p > 0.0 && opts.top_p < 1.0) ? opts.top_p : 1.0;
                int keep = kk;              // nucleus: smallest prefix of the top-kk whose cumsum >= p
                double cum = 0.0;
                for (int i = 0; i < kk; ++i) { cum += x[idx[i]]; if (cum >= p) { keep = i + 1; break; } }
                std::vector<char> kept(static_cast<size_t>(vocab), 0);
                for (int i = 0; i < keep; ++i) kept[idx[i]] = 1;
                double denom2 = 0.0;
                for (int t = 0; t < vocab; ++t) { if (!kept[t]) x[t] = 0.0; denom2 += x[t]; }
                if (denom2 > 0.0) for (int t = 0; t < vocab; ++t) x[t] /= denom2;
            }
            std::uniform_real_distribution<double> u(0.0, 1.0);  // inverse-CDF draw over the (maybe) truncated x
            const double target = u(*opts.rng);
            double acc = 0.0;
            token = vocab - 1;  // fallback for fp rounding at the tail
            for (int t = 0; t < vocab; ++t) { acc += x[t]; if (target <= acc) { token = t; break; } }
            conf = trunc ? orig[token] : x[token];  // the true softmax prob, not the renormalized one
        }
        out.push_back(Candidate{positions[r], token, conf});
    }
    return out;
}

}  // namespace cloze
