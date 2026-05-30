// RNN模型推理
#include "matrix.hpp"
#include "matrix_expr.hpp"
#include "utils.hpp"

namespace utils = ineffa::utils;
namespace console = ineffa::utils::console;
using namespace ineffa::matrix;

auto main(int argc, char* argv[]) -> int try {
    console::init();
    auto args = utils::cmdline::parse(argc, argv);

    // 解析命令行参数
    const std::string model_path = args[0] | "model.bin";
    const float temperature = args["--temperature"] | 0.85f;
    const float top_p = args["--top-p"] | 0.9f;
    const int max_tokens = args["--max-tokens"] | 128;
    const unsigned int random_seed = args["--random-seed"] | std::random_device()();

    // 从文件读取模型的词表和模型参数
    auto file = std::ifstream(std::filesystem::path((char8_t*)model_path.data()), std::ios::binary);
    if (not file)
        throw std::runtime_error("无法打开文件: " + model_path);

    auto read_datas = [&file](auto&&... datas) {
        (file.read((char*)datas.data(), datas.size() * sizeof datas[0]), ...);
    };

    int vocab_size, embed_size, hidden_size;
    file.read((char*)&vocab_size, sizeof vocab_size);
    file.read((char*)&embed_size, sizeof embed_size);
    file.read((char*)&hidden_size, sizeof hidden_size);

    auto vocab = std::vector<char32_t>(vocab_size);
    auto token_to_vocab_index = std::unordered_map<char32_t, int>();

    auto E    = matrix(vocab_size, embed_size);
    auto W_hx = matrix(embed_size, hidden_size);
    auto W_hh = matrix(hidden_size, hidden_size);
    auto W_hy = matrix(hidden_size, vocab_size);
    auto b_h  = matrix<1>(hidden_size);
    auto b_y  = matrix<1>(vocab_size);

    read_datas(vocab, E, W_hx, W_hh, W_hy, b_h, b_y);

    // 生成 token到词表索引的映射
    for (auto [index, token] : vocab | std::ranges::views::enumerate)
        token_to_vocab_index[token] = index;

    // RNN 推理函数
    auto predict = [&](char32_t token, matrix<1>& H, float temperature, float top_p) -> std::pair<char32_t, matrix_view<1>> {
        thread_local auto random_generator = std::mt19937(random_seed);
        thread_local matrix<1> H_next, Y;
        H_next.resize(hidden_size);
        Y.resize(vocab_size);
        
        int token_index = token_to_vocab_index.contains(token) ? token_to_vocab_index[token] : 0;

        matrix_view X = E.slice(token_index);
        H_next = tanh(X * W_hx + H * W_hh + b_h);
        Y = H_next * W_hy + b_y;

        int next_token_index = utils::sample(std::span(Y.data(), Y.size()), random_generator, temperature, top_p);
        char32_t next_token = vocab[next_token_index];
        H = H_next;

        return { next_token, H };
    };

    // 循环从控制台获取提示词输入进行文本生成
    while (true) {
        std::u32string prompt = console::input("输入提示词: ") | utils::utf8_to_utf32;

        auto H = matrix<1>(hidden_size).fill(.0f);
        char32_t token = U' ';

        // 预热阶段：顺着提示词的每一个 token 更新隐藏状态
        for (char32_t prompt_token : prompt) {
            console::print("{}", utils::utf32_to_utf8(prompt_token, true));
            std::tie(token, std::ignore) = predict(prompt_token, H, temperature, top_p);
        }

        // 生成阶段：用预热后的最后一个 token 和 隐藏状态 开始文本生成
        for (int i = 0; i < max_tokens; i++) {
            console::print("{}", utils::utf32_to_utf8(token, true));
            std::tie(token, std::ignore) = predict(token, H, temperature, top_p);
        }
        
        console::print("\n");
    }
}

catch(std::exception& e) {
    console::print("Fatal Error: {}\n", e.what());
    return -1;
}