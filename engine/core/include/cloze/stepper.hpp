// cloze/stepper.hpp — step control (DESIGN §5.3), C++ port of
// lab/cloze_lab/scheduler/stepper.py. Pure logic: owns one decision ("after this
// pass, run another?") plus the per-pass StepContext handed to the policy.
//
// The two Python steppers (FixedStepper / AdaptiveStepper) differ only in whether they
// advertise a step budget, so they collapse into one small value type here. T_max (the
// hard cap) is steps_cap() — the generate loop bounds its pass loop by it, so
// termination is structural even if should_continue always returned true.
#pragma once

#include <stdexcept>

#include "cloze/policies.hpp"

namespace cloze {

// The board-delta of one completed pass, fed back to the controller.
struct StepOutcome {
    int step;             // 0-based index of the pass just run
    int n_committed;      // tokens inked this pass
    int n_masked_after;   // masked positions remaining after the commit
};

struct Stepper {
    enum class Kind { Fixed, Adaptive };
    Kind kind;
    int budget;  // steps (Fixed) or t_max (Adaptive); >= 1

    // fixed(T): exactly `steps` passes (minus the board-drained early exit). Supplies
    // steps_total = steps so ConfidenceTopK quota keeps its per-pass budget.
    static Stepper fixed(int steps) {
        if (steps < 1) throw std::invalid_argument("steps must be >= 1");
        return Stepper{Kind::Fixed, steps};
    }

    // adaptive(T_max): run until the board drains or t_max passes. Supplies no step
    // budget (steps_total = -1), so it composes with threshold and cannot pair with quota.
    static Stepper adaptive(int t_max) {
        if (t_max < 1) throw std::invalid_argument("t_max must be >= 1");
        return Stepper{Kind::Adaptive, t_max};
    }

    // Hard upper bound on passes for one block (the loop's bound).
    int steps_cap() const { return budget; }

    // The StepContext handed to policy.select for this pass.
    StepContext context(int step) const {
        return StepContext{step, kind == Kind::Fixed ? budget : -1};
    }

    // After a pass: true to run another, false to finalize the block.
    bool should_continue(const StepOutcome& o) const { return o.n_masked_after > 0; }
};

}  // namespace cloze
