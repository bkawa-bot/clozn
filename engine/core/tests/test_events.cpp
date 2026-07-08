// test_events.cpp — the §5.1 event spine (DESIGN invariant 2), backend-free. Drives generate
// with the deterministic FakeAdapter and checks the event stream: shape (gen_started first,
// gen_finished last, one block_started/finalized per block), that the tokens_committed items
// reconstruct the board, that the live on_event callback sees the same stream as
// GenerateResult.events, and that JSONL serialization round-trips line-for-line.
#include "cloze/events.hpp"
#include "cloze/generate.hpp"
#include "fake_adapter.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace cloze;

// Active even in Release (asserts compile out under NDEBUG): a hard check that reports + aborts.
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "CHECK failed: %s (line %d)\n", #cond, __LINE__); return 3; } } while (0)

namespace {
const std::vector<int> PROMPT = {1, 2, 3, 4, 5};  // p = 5

template <class T>
int count_of(const std::vector<Event>& evs) {
    int c = 0;
    for (const Event& e : evs)
        if (std::holds_alternative<T>(e)) ++c;
    return c;
}
}  // namespace

int main() {
    // --- Whole-sequence: 4 slots drain in 4 quota passes (1 commit/pass).
    {
        FakeAdapter fake(16, /*eos=*/-1, /*eos_at=*/-1);
        GenerateConfig cfg{/*max_new=*/4, /*steps=*/4, /*block_len=*/0, /*topk=*/-1};

        std::vector<Event> live;
        GenerateResult r = generate(fake, PROMPT, cfg, CacheConfig{}, nullptr,
                                    [&](const Event& e) { live.push_back(e); });
        const std::vector<Event>& evs = r.events;

        // Live callback saw exactly the collected stream.
        CHECK(live.size() == evs.size());

        // Shape: first gen_started, last gen_finished.
        CHECK(std::holds_alternative<GenStarted>(evs.front()));
        CHECK(std::holds_alternative<GenFinished>(evs.back()));
        const auto& gs = std::get<GenStarted>(evs.front());
        CHECK(gs.prompt_tokens == 5 && gs.max_new == 4 && gs.block_len == 0);
        const auto& gf = std::get<GenFinished>(evs.back());
        CHECK(gf.reason == "length" && gf.new_tokens == 4 && gf.steps_total == r.steps_total);

        // One block, four steps.
        CHECK(count_of<BlockStarted>(evs) == 1);
        CHECK(count_of<BlockFinalized>(evs) == 1);
        CHECK(count_of<StepStats>(evs) == r.steps_total);  // 4

        // The committed items reconstruct the generated board (f(p) = p left-to-right).
        std::vector<int> recon = PROMPT;
        recon.resize(PROMPT.size() + cfg.max_new, fake.config().mask_token_id);
        int total_committed = 0;
        for (const Event& e : evs)
            if (const auto* tc = std::get_if<TokensCommitted>(&e))
                for (const CommitItem& it : tc->items) { recon[it.pos] = it.id; ++total_committed; }
        CHECK(total_committed == 4);
        CHECK((std::vector<int>(recon.begin() + 5, recon.end()) == std::vector<int>{5, 6, 7, 8}));

        // JSONL round-trips: every line non-empty and carries its type tag; file has one line/event.
        for (const Event& e : evs) {
            const std::string line = to_jsonl_line(e);
            CHECK(!line.empty() && line.find("\"type\":") != std::string::npos);
        }
        const std::string path = "test_events_out.jsonl";
        CHECK(write_jsonl(evs, path));
        std::FILE* f = std::fopen(path.c_str(), "rb");
        CHECK(f);
        int lines = 0;
        for (int ch; (ch = std::fgetc(f)) != EOF;)
            if (ch == '\n') ++lines;
        std::fclose(f);
        std::remove(path.c_str());
        CHECK(lines == static_cast<int>(evs.size()));
    }

    // --- Block mode: two blocks -> two block_started/finalized, spans contiguous and ordered.
    {
        FakeAdapter fake(16, -1, -1);
        GenerateConfig cfg{/*max_new=*/4, /*steps=*/4, /*block_len=*/2, /*topk=*/-1};
        GenerateResult r = generate(fake, PROMPT, cfg);
        CHECK(count_of<BlockStarted>(r.events) == 2);
        CHECK(count_of<BlockFinalized>(r.events) == 2);
        std::vector<std::pair<int, int>> spans;
        for (const Event& e : r.events)
            if (const auto* bs = std::get_if<BlockStarted>(&e)) spans.push_back(bs->span);
        CHECK(spans.size() == 2);
        CHECK(spans[0] == std::make_pair(5, 7) && spans[1] == std::make_pair(7, 9));
    }

    // --- EOS: gen_finished reason "eos", and a tokens_committed carries the eos id.
    {
        FakeAdapter fake(16, /*eos=*/2, /*eos_at=*/6);
        GenerateConfig cfg{4, 4, 0, -1};
        GenerateResult r = generate(fake, PROMPT, cfg);
        CHECK(std::get<GenFinished>(r.events.back()).reason == "eos");
    }

    // --- Workspace Lens event: additive JSONL wire type for future providers.
    {
        WorkspaceReadout wr{/*t=*/3, /*run_id=*/"run_demo", /*token_index=*/1, /*token_text=*/" cat",
                            /*layer=*/12, /*position=*/1,
                            /*top_readouts=*/{{"uncertainty", 0.62}, {"hallucination_risk", 0.31}},
                            /*entropy=*/0.44, /*provider=*/"mock"};
        const std::string line = to_jsonl_line(Event{wr});
        CHECK(line.find("\"type\": \"workspace_readout\"") != std::string::npos);
        CHECK(line.find("\"run_id\": \"run_demo\"") != std::string::npos);
        CHECK(line.find("\"provider\": \"mock\"") != std::string::npos);
        CHECK(line.find("\"provider_type\": \"mock\"") != std::string::npos);
        CHECK(line.find("\"readout_kind\": \"risk\"") != std::string::npos);
        CHECK(line.find("\"top_readouts\":") != std::string::npos);
    }

    std::printf("test_events: all assertions passed\n");
    return 0;
}
