# core/ — the C++/ggml runtime (Phase 3). See README.md for the full port plan.
#
# Slice 1 (this): the L1 scheduler as a pure-logic library with NO ggml / model / CUDA
# dependency — the same framework-free property the Python scheduler has (DESIGN
# invariant 1). cloze_scheduler must link with no model backend at all; that's what keeps
# the seam honest at the build level and makes the port a translation of
# lab/cloze_lab/scheduler/. The ggml/llama.cpp L0 adapter is a later slice and will be a
# separate library that depends on this one, never the reverse.
cmake_minimum_required(VERSION 3.18)
project(cloze_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cloze_scheduler STATIC
    src/policies.cpp
    src/blocks.cpp
    src/cache.cpp
)
target_include_directories(cloze_scheduler PUBLIC include)

# cloze_runtime: the sampler + the generate pass loop. Still backend-free — it links only
# cloze_scheduler and speaks to the abstract ModelAdapter (model.hpp), so the seam holds at
# the build level (no model backend here either). Mirrors lab/cloze_lab/generate.py, which
# lives above scheduler/ for the same reason. The ggml adapter (below) depends on this; never
# the reverse.
add_library(cloze_runtime STATIC
    src/sample.cpp
    src/selector.cpp
    src/events.cpp
    src/generate.cpp
)
target_link_libraries(cloze_runtime PUBLIC cloze_scheduler)

enable_testing()
foreach(t policies stepper blocks cache)
    add_executable(test_${t} tests/test_${t}.cpp)
    target_link_libraries(test_${t} PRIVATE cloze_scheduler)
    add_test(NAME ${t} COMMAND test_${t})
endforeach()

# The generate pass loop + the event spine, exercised in the backend-free build via a
# deterministic FakeAdapter (no ggml). Link cloze_runtime (generate + sample + events).
add_executable(test_generate tests/test_generate.cpp)
target_link_libraries(test_generate PRIVATE cloze_runtime)
add_test(NAME generate COMMAND test_generate)

add_executable(test_events tests/test_events.cpp)
target_link_libraries(test_events PRIVATE cloze_runtime)
add_test(NAME events COMMAND test_events)

add_executable(test_infill tests/test_infill.cpp)
target_link_libraries(test_infill PRIVATE cloze_runtime)
add_test(NAME infill COMMAND test_infill)

add_executable(test_revision tests/test_revision.cpp)
target_link_libraries(test_revision PRIVATE cloze_runtime)
add_test(NAME revision COMMAND test_revision)

# GAP #1 (task #43): the white-box state WRITE seam (ModelAdapter::write_state) — the
# read -> edit -> write -> observe loop, proven against a writable test double (backend-free, CPU).
# Header-only against the seam; links cloze_scheduler for the include dir. The ggml adapter implements
# write_state against the live llama context in a later slice; this fixes the contract first.
add_executable(test_state_write tests/test_state_write.cpp)
target_link_libraries(test_state_write PRIVATE cloze_scheduler)
add_test(NAME state_write COMMAND test_state_write)

# --- Confidence-select kernel selector (Phase 2 kernel wired into the §4.3 seam) ---
# Optional + needs a CUDA compiler: builds the vendored confidence-select kernel and the
# KernelCommitSelector that drives it from the generate loop, plus a parity test that gates it
# against the backend-free CpuCommitSelector. Default OFF so the scheduler/runtime build (and
# CUDA-less CI) stay pure C++. The kernel's greedy/MaxProb/TopK path is validated vs
# reference.py on sm_120; this proves it drops into the runtime with identical picks.
#
#   cmake -S core -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZE_BUILD_CUDA=ON
#   cmake --build build-cuda --target test_kernel_selector && ctest -R kernel_selector
#
# Windows note: use the Ninja generator (it drives nvcc directly), inside a VS dev shell
# (vcvars64) with nvcc on PATH. The Visual Studio generator needs CUDA's MSBuild integration
# installed into VS, which the CUDA-toolkit-only setup here lacks — Ninja sidesteps that.
# Validated on an RTX 5080 (sm_120, CUDA 13.3): test_kernel_selector PASSES (5/5 parity cases).
option(CLOZE_BUILD_CUDA "Build the CUDA confidence-select selector + parity test" OFF)

# --- On-device SAE readout (ROADMAP 3.3 wired into the engine) ---
# cloze_sae (src/sae_encoder.cu + cloze/sae.hpp): the JumpReLU SAE encoder GEMV + the vendored
# sae_topk kernel, behind a CUDA-free pimpl header so cloze-server links it untouched by nvcc.
# Weights come from tools/export_sae_weights.py; parity vs the torch oracle is gated by
# test_sae_encoder against tools/dump_sae_vectors.py artifacts (SKIPs where the export is absent).
#   cmake -S core -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZE_BUILD_SAE=ON
option(CLOZE_BUILD_SAE "Build the on-device SAE readout (JumpReLU encoder + sae_topk, CUDA)" OFF)

if(CLOZE_BUILD_CUDA OR CLOZE_BUILD_SAE)
    # Ampere/Ada/Blackwell for all CUDA targets — set BEFORE enable_language (CMake otherwise
    # pre-seeds a default at enable time), matching kernels/confidence_select/CMakeLists.txt.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;89;120")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # SAE-sparsify top-k kernel (ROADMAP 3.3: the confidence-select top-k repointed at the FEATURE
    # dim), vendored from kernels/ — shared by the CUDA parity tests AND the SAE readout library.
    # Validated on sm_120: selected indices exact, values within f32-vs-f64 eps (also proven
    # standalone by kernels/sae_topk/validate.py vs the numpy reference).
    set(SAE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/sae_topk)  # monorepo: engine/core -> kernels/
    add_library(cloze_sae_topk STATIC ${SAE_DIR}/sae_topk.cu)
    set_target_properties(cloze_sae_topk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(cloze_sae_topk PUBLIC ${SAE_DIR})
endif()

if(CLOZE_BUILD_CUDA)
    set(CS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/confidence_select)  # monorepo: engine/core -> kernels/
    add_library(cloze_confidence_select STATIC ${CS_DIR}/confidence_select.cu)
    set_target_properties(cloze_confidence_select PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(cloze_confidence_select PUBLIC ${CS_DIR})

    add_library(cloze_cuda STATIC src/selector_kernel.cu)
    set_target_properties(cloze_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(cloze_cuda PUBLIC cloze_runtime cloze_confidence_select)

    add_executable(test_kernel_selector tests/test_kernel_selector.cu)
    set_target_properties(test_kernel_selector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_kernel_selector PRIVATE cloze_cuda)
    add_test(NAME kernel_selector COMMAND test_kernel_selector)

    # Self-contained C++ parity test gating the sae_topk kernel (built above, shared with
    # CLOZE_BUILD_SAE) against an in-process CPU reference.
    add_executable(test_sae_topk tests/test_sae_topk.cu)
    set_target_properties(test_sae_topk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_sae_topk PRIVATE cloze_sae_topk)
    add_test(NAME sae_topk COMMAND test_sae_topk)
endif()

if(CLOZE_BUILD_SAE)
    find_package(CUDAToolkit REQUIRED)
    # The SAE readout library: JumpReLU encoder GEMV + epilogue in front of the sae_topk kernel.
    # CUDA_RESOLVE_DEVICE_SYMBOLS so plain-CXX consumers (cloze-server, test_sae_encoder) link it
    # without a device-link step of their own; cudart_static rides the public interface for the same
    # reason. See cloze/sae.hpp for the pipeline + numerics contract.
    add_library(cloze_sae STATIC src/sae_encoder.cu)
    set_target_properties(cloze_sae PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_include_directories(cloze_sae PUBLIC include)
    target_link_libraries(cloze_sae PUBLIC cloze_sae_topk CUDA::cudart_static)

    # Engine-vs-torch parity gate (SKIPs cleanly when the exported weights are absent, so CI boxes
    # without the 0.94 GB export stay green — same convention as test_ggml_state_write).
    add_executable(test_sae_encoder tests/test_sae_encoder.cpp)
    target_link_libraries(test_sae_encoder PRIVATE cloze_sae)
    add_test(NAME sae_encoder COMMAND test_sae_encoder)
endif()

# --- L0 ggml adapter (Phase 3, slice 3) ---
# Optional + heavy: this builds the vendored llama.cpp. Default OFF so the scheduler-only
# build (and CI) stay fast and dependency-free — the scheduler library above never links a
# model backend (the seam, enforced at the build level). The diffusion forward is validated
# against the lab by test_ggml_forward: it loads a GGUF, runs a bidirectional decode through
# llama.cpp (llama_set_causal_attn false), applies the Dream-family shift, and the first
# masked slot of "...return a +" predicts ' b' (token 293) — matching test_dcoder_adapter.py.
#
#   cmake -S core -B build-ggml -DCLOZE_BUILD_GGML=ON -DGGML_CUDA=OFF
#   cmake --build build-ggml --target test_ggml_forward
#   ./build-ggml/.../test_ggml_forward  <model.gguf>   (convert via llama.cpp/convert_hf_to_gguf.py)
option(CLOZE_BUILD_GGML "Build the ggml L0 adapter + forward test (builds llama.cpp)" OFF)
if(CLOZE_BUILD_GGML)
    add_subdirectory(third_party/llama.cpp)

    # GgmlAdapter: the ONE place a model backend lives (DESIGN invariant 1). It links
    # cloze_runtime (the loop + sampler) and llama; the scheduler/runtime libs never link
    # llama, so the backend-free build stays backend-free.
    # generate_ar.cpp (the autoregressive white-box loop) lives here, not in cloze_runtime: it drives
    # GgmlAdapter::ar_forward directly (incremental causal decode), so it's backend-coupled by nature —
    # unlike the diffusion generate()/infill()/denoise(), which stay abstract-adapter-only (the seam).
    add_library(cloze_ggml STATIC src/model_ggml.cpp src/generate_ar.cpp)
    target_link_libraries(cloze_ggml PUBLIC cloze_runtime llama)

    # Raw forward smoke test (no scheduler): proves the bidirectional decode + Dream shift.
    add_executable(test_ggml_forward tests/test_ggml_forward.cpp)
    target_link_libraries(test_ggml_forward PRIVATE llama)

    # End-to-end slice 1: drive the generate loop through the seam, cross-check vs the lab.
    add_executable(test_ggml_generate tests/test_ggml_generate.cpp)
    target_link_libraries(test_ggml_generate PRIVATE cloze_ggml)

    # GAP #1 (task #43): the white-box state-WRITE on a REAL model — read -> edit -> write -> observe ->
    # clear through llama, via the patch-free eval-callback activation patch. SKIPS without a GGUF arg.
    add_executable(test_ggml_state_write tests/test_ggml_state_write.cpp)
    target_link_libraries(test_ggml_state_write PRIVATE cloze_ggml)
    if(WIN32)
        # llama/ggml DLLs are emitted under build/bin.  The product launcher prepends that directory,
        # but a bare CTest invocation otherwise stalls behind Windows' missing-DLL error dialog before
        # this test can reach its intentional no-model skip.
        add_test(
            NAME ggml_state_write
            COMMAND ${CMAKE_COMMAND} -E env
                "PATH=$<TARGET_FILE_DIR:llama>;$ENV{PATH}"
                $<TARGET_FILE:test_ggml_state_write>
        )
    else()
        add_test(NAME ggml_state_write COMMAND test_ggml_state_write)
    endif()

    # cloze — the native runtime CLI: load a GGUF, denoise a prompt, print the completion + stats.
    # The "it actually runs" artifact (the daily-driver shape behind the tests).
    add_executable(cloze tools/cloze_cli.cpp)
    target_link_libraries(cloze PRIVATE cloze_ggml)

    # cloze-bench — the §9 quant-calibration sweep: quantize f16 -> Q8/Q5/Q4 in-process and
    # report size + token-agreement-vs-f16 + tok/s (the honest quality column for the quant axis).
    add_executable(cloze-bench tools/cloze_bench.cpp)
    target_link_libraries(cloze-bench PRIVATE cloze_ggml)

    # cloze-arbench — autoregressive vs diffusion decode THROUGHPUT on the same weights/GPU.
    add_executable(cloze-arbench tools/cloze_arbench.cpp)
    target_link_libraries(cloze-arbench PRIVATE cloze_ggml)

    # cloze-probe-sweep — sweep all layers to find the best tap for concept probes.
    add_executable(cloze-probe-sweep tools/cloze_probe_sweep.cpp)
    target_link_libraries(cloze-probe-sweep PRIVATE cloze_ggml)

    # cloze-ar — autoregressive white-box CLI: read tap + logit-lens per token on any AR GGUF.
    add_executable(cloze-ar tools/cloze_ar.cpp)
    target_link_libraries(cloze-ar PRIVATE cloze_ggml)

    # Resolving experiment for the zero-copy §4.3 path: does llama_get_logits_tensor (the CLOZE
    # PATCH accessor) return a DEVICE-resident tensor that byte-matches the host logits? Needs a
    # CUDA-capable llama (configure with -DGGML_CUDA=ON) and runs with all layers offloaded.
    add_executable(test_device_logits tests/test_device_logits.cpp)
    target_link_libraries(test_device_logits PRIVATE llama)
endif()

# --- Zero-copy §4.3 end-to-end (needs BOTH the ggml adapter AND the CUDA kernel selector) ---
# Three-way parity: CpuCommitSelector vs KernelCommitSelector(host-upload) vs
# KernelCommitSelector(device zero-copy) must all reproduce the lab golden. Build with a CUDA
# llama (-DGGML_CUDA=ON) so the device path engages.
#   cmake -S core -B build-zc -G Ninja -DCLOZE_BUILD_GGML=ON -DGGML_CUDA=ON -DCLOZE_BUILD_CUDA=ON
if(CLOZE_BUILD_GGML AND CLOZE_BUILD_CUDA)
    add_executable(test_zerocopy tests/test_zerocopy.cpp)
    target_link_libraries(test_zerocopy PRIVATE cloze_ggml cloze_cuda)
endif()

# --- L2 serving layer (HTTP, needs CLOZE_BUILD_GGML) ---
# An HTTP server over the runtime: /v1/completions (+ SSE streaming of the §5.1 events) and
# /v1/infill, using the cpp-httplib + nlohmann/json that llama.cpp already vendors (no new deps).
#   cmake -S core -B build -G Ninja -DCLOZE_BUILD_GGML=ON -DCLOZE_BUILD_SERVE=ON
option(CLOZE_BUILD_SERVE "Build the HTTP serving layer (requires CLOZE_BUILD_GGML)" OFF)
if(CLOZE_BUILD_SERVE AND CLOZE_BUILD_GGML)
    set(_vendor ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
    # cpp-httplib is the split build: compile its httplib.cpp here for the out-of-line definitions
    # (silence its 3rd-party warnings; it needs /bigobj on MSVC).
    # Phase 12.4: the serve monolith split into server_main + route-family TUs that read shared state
    # through ServerContext (serve/server_context.hpp) + the shared helpers/structs (serve/server_shared.hpp).
    add_executable(cloze-server
        serve/server_main.cpp
        serve/routes_jlens.cpp
        serve/routes_whitebox.cpp
        serve/routes_state.cpp
        ${_vendor}/cpp-httplib/httplib.cpp)
    target_include_directories(cloze-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/serve ${_vendor}/cpp-httplib ${_vendor})
    target_link_libraries(cloze-server PRIVATE cloze_ggml)
    if(CLOZE_BUILD_SAE)
        # --sae <dir>: the on-device SAE feature readout rides the server (cloze/sae.hpp is a
        # CUDA-free pimpl header, so this stays a plain-CXX target; cloze_sae brings cudart).
        target_link_libraries(cloze-server PRIVATE cloze_sae)
        target_compile_definitions(cloze-server PRIVATE CLOZE_SAE)
    endif()
    find_package(Threads REQUIRED)
    target_link_libraries(cloze-server PRIVATE Threads::Threads)
    if(WIN32)
        target_link_libraries(cloze-server PRIVATE ws2_32)
    endif()
    if(MSVC)
        # /EHsc: enable C++ unwind semantics so a thrown exception (e.g. nlohmann json::value() on a
        # wrong-typed field in a malformed request body) UNWINDS into httplib's handler-exception
        # catch (-> 500) instead of crashing the process. ggml's global flags leave it off (the
        # C4530 warning), which turned a bad request into a server-killing DoS. /bigobj: httplib needs it.
        target_compile_options(cloze-server PRIVATE /bigobj /EHsc)
        set_source_files_properties(${_vendor}/cpp-httplib/httplib.cpp PROPERTIES COMPILE_FLAGS /w)
    endif()
endif()
