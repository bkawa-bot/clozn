// test_stepper.cpp — checks for the C++ stepper port, mirroring test_stepper.py.
#include "cloze/stepper.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>

using namespace cloze;

int main() {
    // fixed(T): advertises the budget so quota mode keeps it; caps at T.
    {
        auto s = Stepper::fixed(4);
        assert(s.steps_cap() == 4);
        assert(s.context(0).steps_total == 4);
        assert(s.context(0).steps_remaining() == 4);
        assert(s.context(3).steps_remaining() == 1);
        assert(s.should_continue(StepOutcome{0, 1, 2}) == true);   // masks remain
        assert(s.should_continue(StepOutcome{3, 2, 0}) == false);  // drained
    }
    // adaptive(T_max): no step budget (steps_total = -1), caps at t_max.
    {
        auto s = Stepper::adaptive(8);
        assert(s.steps_cap() == 8);
        assert(s.context(0).steps_total == -1);
        assert(s.context(0).steps_remaining() == -1);
        assert(s.should_continue(StepOutcome{0, 1, 3}) == true);
    }
    // validation: budgets must be >= 1.
    {
        bool threw = false;
        try { Stepper::fixed(0); } catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
        threw = false;
        try { Stepper::adaptive(0); } catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    }
    std::printf("test_stepper: all assertions passed\n");
    return 0;
}
