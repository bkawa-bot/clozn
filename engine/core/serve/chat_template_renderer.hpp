#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_model;

namespace cloze {

// Keeps llama-common's Jinja/chat types out of the worker's public headers. In particular,
// chat.h defines its own global JSON alias, which must not bleed into server_shared.hpp.
class ChatTemplateRenderer {
public:
    explicit ChatTemplateRenderer(const llama_model* model);
    ~ChatTemplateRenderer();

    ChatTemplateRenderer(const ChatTemplateRenderer&) = delete;
    ChatTemplateRenderer& operator=(const ChatTemplateRenderer&) = delete;

    bool available() const noexcept;
    std::string apply(const std::vector<std::pair<std::string, std::string>>& messages,
                      bool add_assistant) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cloze
