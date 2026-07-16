#include "chat_template_renderer.hpp"

#include "chat.h"

#include <stdexcept>

namespace cloze {

struct ChatTemplateRenderer::Impl {
    common_chat_templates_ptr templates;
};

ChatTemplateRenderer::ChatTemplateRenderer(const llama_model* model) : impl_(std::make_unique<Impl>()) {
    if (model != nullptr && llama_model_chat_template(model, /*name=*/nullptr) != nullptr) {
        impl_->templates = common_chat_templates_init(model, /*chat_template_override=*/"");
    }
}

ChatTemplateRenderer::~ChatTemplateRenderer() = default;

bool ChatTemplateRenderer::available() const noexcept {
    return impl_ != nullptr && impl_->templates != nullptr;
}

std::string ChatTemplateRenderer::apply(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_assistant) const {
    if (!available()) {
        throw std::runtime_error("model has no embedded chat template");
    }
    common_chat_templates_inputs inputs;
    inputs.messages.reserve(messages.size());
    for (const auto& message : messages) {
        inputs.messages.push_back(common_chat_msg{message.first, message.second});
    }
    inputs.add_generation_prompt = add_assistant;
    inputs.use_jinja = true;
    return common_chat_templates_apply(impl_->templates.get(), inputs).prompt;
}

}  // namespace cloze
