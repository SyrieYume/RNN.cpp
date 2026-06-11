// RNN模型训练
#include <atomic>
#include <csignal>
#include "matrix.hpp"
#include "matrix_expr.hpp"
#include "utils.hpp"

using namespace ineffa::matrix;
namespace utils = ineffa::utils;
namespace console = ineffa::utils::console;

constexpr int embed_size = 64;     // 词向量的维度 (必须是16的倍数)
constexpr int hidden_size = 512;   // 隐藏层大小 (必须是16的倍数)
constexpr int batch_size = 32;     // 训练时的批大小
constexpr int vocab_size = 3200;   // 词表大小 (必须是16的倍数)

constexpr int seq_length = 42;          // 训练时的反向传播的时间步长
constexpr float learning_rate = 0.002f; // 训练时的学习率
constexpr int train_steps = 5000;       // 训练的步数
constexpr int log_interval = 200;       // 训练时每隔多少步打印一次输出

uint32_t random_seed = std::random_device()();      // 随机数种子 (生成随机种子: std::random_device()())
auto random_generator = std::mt19937(random_seed);  // 随机数生成器
std::atomic<bool> stop_requested = false;           // 控制台中断信号

// 词表 和 token到词表索引的映射
auto vocab = std::vector<char32_t>();
auto token_to_vocab_index = std::unordered_map<char32_t, int>();

// 模型参数
auto E    = matrix<2>(vocab_size, embed_size).fill_random(random_generator, 0.01f);
auto W_hx = matrix<2>(embed_size, hidden_size).fill_random(random_generator, 0.01f);
auto W_hh = matrix<2>(hidden_size, hidden_size).fill_random(random_generator, 0.01f);
auto W_hy = matrix<2>(hidden_size, vocab_size).fill_random(random_generator, 0.01f);
auto b_h  = matrix<1>(hidden_size).fill(.0f);
auto b_y  = matrix<1>(vocab_size).fill(.0f);

// 上一个训练步的最后一个 RNN 隐藏状态
auto H_previous = matrix<2>(batch_size, hidden_size).fill(.0f);

// 前向传播计算的中间结果
auto Xs = matrix<3>(seq_length, batch_size, embed_size);
auto Hs = matrix<3>(seq_length, batch_size, hidden_size);
auto Ys = matrix<3>(seq_length, batch_size, vocab_size);
auto Ps = matrix<3>(seq_length, batch_size, vocab_size);

// 反向传播中计算的梯度
auto dE    = matrix<2>(vocab_size, embed_size);
auto dW_hx = matrix<2>(embed_size, hidden_size);
auto dW_hh = matrix<2>(hidden_size, hidden_size);
auto dW_hy = matrix<2>(hidden_size, vocab_size);
auto db_h  = matrix<1>(hidden_size);
auto db_y  = matrix<1>(vocab_size);

auto dY = matrix<2>(batch_size, vocab_size);
auto dH = matrix<2>(batch_size, hidden_size);
auto dZ = matrix<2>(batch_size, hidden_size);
auto dZ_next = matrix<2>(batch_size, hidden_size);
auto dX = matrix<2>(batch_size, embed_size);

// RMSprop 优化器的内存变量
auto mE    = matrix<2>(vocab_size, embed_size).fill(.0f);
auto mW_hx = matrix<2>(embed_size, hidden_size).fill(.0f);
auto mW_hh = matrix<2>(hidden_size, hidden_size).fill(.0f);
auto mW_hy = matrix<2>(hidden_size, vocab_size).fill(.0f);
auto mb_h  = matrix<1>(hidden_size).fill(.0f);
auto mb_y  = matrix<1>(vocab_size).fill(.0f);


// 前向传播 (计算交叉熵损失)
auto rnn_forward(std::span<const int> inputs, std::span<const int> targets) noexcept -> float {
    float loss = 0.0f;

    for (int t : std::views::iota(0, seq_length)) {
        matrix_view X = Xs.slice(t);
        matrix_view H = Hs.slice(t);
        matrix_view Y = Ys.slice(t);
        matrix_view P = Ps.slice(t);
        std::span input = inputs.subspan(t * batch_size, batch_size);
        std::span target = targets.subspan(t * batch_size, batch_size);

        // Embedding查表 : X[batch] = E[token_index]
        for (int batch = 0; batch < batch_size; batch++)
            X.slice(batch) = E.slice(input[batch]).copy();

        // 获取上一个时间步的隐藏状态
        matrix_view H_previous = t == 0 ? ::H_previous : Hs.slice(t - 1);

        // RNN 核心
        H = tanh(X * W_hx + H_previous * W_hh + b_h);

        // 计算输出层
        Y = H * W_hy + b_y;
        
        // Softmax 归一化
        P = softmax(Y);

        // 累加交叉熵损失：loss = sum(-log(P[target]), axis=batch) / batch_size
        loss += cross_entropy_loss(P, target);
    }

    return loss / seq_length;
}


// 反向传播 (计算模型各个参数的梯度)
void rnn_backward(std::span<const int> inputs, std::span<const int> targets) noexcept {
    dE.fill(.0f);
    dW_hx.fill(.0f);
    dW_hh.fill(.0f);
    db_h.fill(.0f);
    dW_hy.fill(.0f);
    db_y.fill(.0f);
    dZ_next.fill(.0f);

    for (int t : std::views::iota(0, seq_length) | std::views::reverse) {
        matrix_view X = Xs.slice(t);
        matrix_view H = Hs.slice(t);
        matrix_view P = Ps.slice(t);
        std::span input = inputs.subspan(t * batch_size, batch_size);
        std::span target = targets.subspan(t * batch_size, batch_size);

        // dY = (P - one_hot(targets)) / (batch_size * seq_length)
        for (int batch = 0; batch < batch_size; batch++)
            P[batch, target[batch]] -= 1.0f;

        for (int i : std::views::iota(0, P.size()))
            dY[i] = P[i] / (batch_size * seq_length);

        dH = dZ_next * W_hh.T() + dY * W_hy.T();
        dW_hy += H.T() * dY;
        db_y += sum(dY /* axis=batch */);
        
        // dZ = dH * tan'(H) = dH * (1 - H * H)
        for (int batch = 0; batch < batch_size; batch++) {
            const float* __restrict dH_row = dH.data() + batch * hidden_size;
            const float* __restrict H_row = H.data() + batch * hidden_size;
            float* __restrict dZ_row = dZ.data() + batch * hidden_size;

            for (int h = 0; h < hidden_size; h++)
                dZ_row[h] = dH_row[h] * (1.0f - H_row[h] * H_row[h]);
        }

        // 获取上一个时间步的隐藏状态
        matrix_view<2> H_previous = t == 0 ? ::H_previous : Hs.slice(t - 1);

        dW_hh += H_previous.T() * dZ;
        dW_hx += X.T() * dZ;
        db_h += sum(dZ /* axis=batch */);
        dX = dZ * W_hx.T();
        
        // 累加 Embedding 层的梯度: dE[input] += dX
        for (int batch = 0; batch < batch_size; batch++)
            for (int e = 0; e < embed_size; e++)
                dE[input[batch], e] += dX[batch, e];

        dZ_next = dZ;
    }
}


// 使用反向传播计算出的梯度，通过优化器更新模型参数
void rnn_optimize() noexcept {
    // RMSprop 优化器
    auto apply_rmsprop = [](auto& params, auto& mems, auto& grads) {
        constexpr float alpha = 0.95f;
        const int length = params.size();

        #pragma omp simd
        for (int i = 0; i < length; i++) {
            const float grad = std::clamp(grads[i], -5.0f, 5.0f);
            mems[i] = alpha * mems[i] + (1.0f - alpha) * grad * grad;
            params[i] -= learning_rate * grad / std::sqrt(mems[i] + 1e-8f);
        }
    };
    
    apply_rmsprop(E, mE, dE);
    apply_rmsprop(W_hx, mW_hx, dW_hx);
    apply_rmsprop(W_hh, mW_hh, dW_hh);
    apply_rmsprop(W_hy, mW_hy, dW_hy);
    apply_rmsprop(b_h, mb_h, db_h);
    apply_rmsprop(b_y, mb_y, db_y);
}


// RNN 推理函数
auto predict(char32_t token, matrix<1>& H, float temperature, float top_p) -> std::pair<char32_t, matrix_view<1>> {
    thread_local matrix<1> H_next, Y;

    H_next.resize(hidden_size);
    Y.resize(vocab_size);
    
    int token_index = token_to_vocab_index.contains(token) ? token_to_vocab_index[token] : 0;

    matrix_view X = E.slice(token_index);
    H_next = tanh(X * W_hx + H * W_hh + b_h);
    Y = H_next * W_hy + b_y;
    H = H_next;

    int next_token_index = utils::sample(std::span(Y.data(), Y.size()), random_generator, temperature, top_p);
    char32_t next_token = vocab[next_token_index];

    return { next_token, H };
}


auto main(int argc, char* argv[]) -> int try {
    console::init();
    auto args = utils::cmdline::parse(argc, argv);
    
    console::info("训练参数: seed={} | dim={}/{}/{} | lr={} | bs={} | seq={} | steps={}", 
        random_seed, vocab_size, embed_size, hidden_size, learning_rate, batch_size, seq_length, train_steps);

    // 解析命令行参数
    std::string data_path = args[0] | "data.txt";
    std::string output_path = args["-o"] | args["--output"] | "model.bin";

    console::info("加载训练数据: {}", data_path);
    std::u32string data = utils::load_files(data_path) | utils::utf8_to_utf32;

    if (data.size() < batch_size * (seq_length + 1))
        throw std::runtime_error("训练数据量过少");

    std::tie(vocab, token_to_vocab_index) = utils::generate_vocab(data, vocab_size);
    
    // 拦截终端的中断信号 (Ctrl+C)
    std::signal(SIGINT, [](int signal) {
        stop_requested = true;
    });

    auto inputs  = std::vector<int>(seq_length * batch_size);
    auto targets = std::vector<int>(seq_length * batch_size);

    float smooth_loss = -std::log(1.0f / vocab_size);
    
    // 将训练数据分成 batch_size 个 chunk，每个 batch 在各自对应 chunk 的数据上从上到下训练
    const int chunk_size = data.size() / batch_size;

    // 训练循环
    for (int step = 0, chunk_index = 0; step < train_steps; step++, chunk_index += seq_length) {
        if (stop_requested) {
            console::info("检测到中断信号 (Ctrl+C)，正在停止训练并保存模型...");
            break;
        }

        if (chunk_index + seq_length + 1 >= chunk_size) {
            chunk_index = 0;
            H_previous.fill(.0f); 
        }

        for (int b = 0; b < batch_size; b++)
            for (int t = 0; t < seq_length; t++) {
                inputs[t * batch_size + b] = token_to_vocab_index[data[b * chunk_size + chunk_index + t]];
                targets[t * batch_size + b] = token_to_vocab_index[data[b * chunk_size + chunk_index + t + 1]];
            }

        float loss = rnn_forward(inputs, targets);
        rnn_backward(inputs, targets);
        rnn_optimize();

        smooth_loss = smooth_loss * 0.95f + loss * 0.05f;

        if (step % log_interval == 0) {
            console::info("Step: {} | Loss: {:>.3f} | Smooth Loss: {:>.3f}", step, loss, smooth_loss);
            console::print("测试生成: ");

            auto H_test = matrix<1>(hidden_size).fill(.0f);
            char32_t token = U'\n';

            for (int i = 0; i < seq_length; i++) {
                std::tie(token, std::ignore) = predict(token, H_test, 0.85f, 0.9f);
                console::print("{}", utils::utf32_to_utf8(token, true));
            }
            console::print("\n");
        }
    
        // 将当前序列最后一个时间步的隐藏状态保存，供下一步的训练使用
        H_previous = Hs.slice(seq_length - 1).copy();
    }


    // 将训练好的模型保存到文件
    auto file = std::ofstream(std::filesystem::path((char8_t*)output_path.data()), std::ios::binary);
    if (not file)
        throw std::runtime_error(std::format("无法打开文件 '{}'", output_path));

    auto write_datas = [&file](auto&&... datas) {
        (file.write((char*)datas.data(), datas.size() * sizeof(datas[0])), ...);
    };

    write_datas(std::array { vocab_size, embed_size, hidden_size });
    write_datas(vocab, E, W_hx, W_hh, W_hy, b_h, b_y);

    console::info("模型已保存至: {}", output_path);
}

catch(std::exception& e) {
    console::print("Fatal Error: {}\n", e.what());
    return -1;
}