#pragma once
#include "matrix.hpp"
#include <cmath>

namespace ineffa::matrix {

// Y = X * W + b (向量乘矩阵)
// Y: [M], X: [K], W: [K, M], b: [M]
inline void mat_mul_add(matrix_view<1> Y, const matrix_view<1> X, const matrix_view<> W, const matrix_view<1> b) noexcept {
    const int M = W.extent<1>();
    const int K = W.extent<0>();

    for (int j = 0; j < M; j++)
        Y[j] = b[j];

    for (int k = 0; k < K; k++) {
        const float x = X[k];
        const float* __restrict W_row = W.data() + k * M;

        for (int j = 0; j < M; j++)
            Y[j] += x * W_row[j];
    }
}


// Y = X * W + b (矩阵乘矩阵)
// Y: [N, M], X: [N, K], W: [K, M], b: [M]
inline void mat_mul_add(matrix_view<> Y, const matrix_view<> X, const matrix_view<> W, const matrix_view<1> b) {
    const int N = Y.extent<0>();
    const int M = Y.extent<1>();
    const int K = X.extent<1>(); 

    #pragma omp parallel for if(M * K * N >= 3276800)
    for (int i = 0; i < N; i++) {
        float* __restrict Y_row = Y.data() + i * M;
        const float* __restrict X_row = X.data() + i * K;

        // Y = b
        for (int j = 0; j < M; j++)
            Y_row[j] = b[j];

        // Y += X * W
        for (int k = 0; k < K; k++) {
            const float x = X_row[k];
            const float* __restrict W_row = W.data() + k * M;
            
            for (int j = 0; j < M; j++)
                Y_row[j] += x * W_row[j];
        }
    }
}


// RNN 前向传播公式 (X为矩阵)
// H_next = tanh(X * W_hx + H * W_hh + b_h)
// H_next: [N, M], X: [N, K], W_hx: [K, M], H: [N, M], W_hh: [M, M], b_h: [M]
inline void mat_rnn_forward(matrix_view<> H_next, const matrix_view<> X, const matrix_view<> W_hx, const matrix_view<> H, const matrix_view<> W_hh, const matrix_view<1> b_h) noexcept {
    const int N = X.extent<0>();
    const int M = H_next.extent<1>();
    const int K = X.extent<1>();

    #pragma omp parallel for if(M * N * (K + M) >= 3276800)
    for (int i = 0; i < N; i++) {
        float* __restrict H_next_row = H_next.data() + i * M;
        const float* __restrict X_row = X.data() + i * K;
        const float* __restrict H_row = H.data() + i * M;

        // H_next = b
        for (int j = 0; j < M; j++) 
            H_next_row[j] = b_h[j];

        // H_next += X * W_hx
        for (int k = 0; k < K; k++) {
            const float x = X_row[k];
            const float* __restrict W_hx_row = W_hx.data() + k * M;

            for (int j = 0; j < M; j++) 
                H_next_row[j] += x * W_hx_row[j];
        }

        // H_next += H * W_hh
        for (int k = 0; k < M; k++) {
            const float h = H_row[k];
            const float* __restrict W_hh_row = W_hh.data() + k * M;
            
            for (int j = 0; j < M; j++) 
                H_next_row[j] += h * W_hh_row[j];
        }
        
        // H_next = tanh(H_next)
        for (int j = 0; j < M; j++) 
            H_next_row[j] = std::tanh(H_next_row[j]);
    }
}


// RNN 前向传播公式 (X为向量)
// H_next = tanh(X * W_hx + H * W_hh + b_h)
// H_next: [M], X: [K], W_hx: [K, M], H: [N, M], W_hh: [M, M], b_h: [M]
inline void mat_rnn_forward(matrix_view<1> H_next, matrix_view<1> X, const matrix_view<> W_hx, const matrix_view<1> H, const matrix_view<> W_hh, const matrix_view<1> b_h) noexcept {
    matrix_view H_next_ = H_next.reshape(1, H_next.size());
    matrix_view H_ = H.reshape(1, H.size());
    matrix_view X_ = X.reshape(1, X.size());
    mat_rnn_forward(H_next_, X_, W_hx, H_, W_hh, b_h);
}


// 计算 Softmax: P = exp(Y - max(Y, axis=M)) / sum(exp(Y - max(Y, axis=M)), axis=M)
// P: [N, M], Y: [N, M]
inline void mat_softmax(matrix_view<> P, const matrix_view<> Y) {
    const int N = Y.extent<0>();
    const int M = Y.extent<1>();

    for (int i = 0; i < N; i++) {
        const float* __restrict Y_row = Y.data() + i * M;
        float* __restrict P_row = P.data() + i * M;
        
        // max_val = max(Y, axis=M)
        float max_val = std::numeric_limits<float>::lowest();
        for (int j = 0; j < M; j++)
            max_val = std::max(max_val, Y_row[j]);

        // P = exp(Y - max_val)
        // sum_exp = sum(P, axis=M)
        float sum_exp = 0.0f;
        #pragma omp simd reduction(+:sum_exp)
        for (int j = 0; j < M; j++) {
            const float exp_val = std::exp(Y_row[j] - max_val);
            P_row[j] = exp_val;
            sum_exp += exp_val;
        }

        // 归一化 : P = P / sum_exp
        const float inv_sum_exp = 1.0f / sum_exp;
        for (int j = 0; j < M; j++)
            P_row[j] *= inv_sum_exp;
    }
}


// 计算交叉熵损失 loss = sum(-log(P[target]), axis=N) / N
// P: [N, M], target: [N]
inline auto cross_entropy_loss(const matrix_view<> P, const std::span<const int> targets_idx) -> float {
    const int N = P.extent<0>();
    const int M = P.extent<1>();

    float total_loss = 0.0f;
    constexpr float epsilon = 1e-7f;

    for (int i = 0; i < N; i++) {
        const float prob_target = P[i * M + targets_idx[i]]; 
        total_loss -= std::log(prob_target + epsilon);
    }

    return total_loss / N;
}


// Y = A * B^T
// Y: [N, M], A: [N, K], B: [M, K]
inline void mat_mul_A_Tb(matrix_view<> Y, const matrix_view<> A, const matrix_view<> B) noexcept {
    const int N = Y.extent<0>();
    const int M = Y.extent<1>(); 
    const int K = A.extent<1>();

    #pragma omp parallel for if(M * K * N >= 3276800)
    for (int i = 0; i < N; i++) {
        float* __restrict Y_row = Y.data() + i * M;
        const float* __restrict A_row = A.data() + i * K;

        for (int j = 0; j < M; j++) {
            const float* __restrict B_row = B.data() + j * K;
            
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++)
                sum += A_row[k] * B_row[k];

            Y_row[j] = sum;
        }
    }
}


// Y += A * B^T
// Y: [N, M], A: [N, K], B: [M, K]
inline void mat_mul_add_A_Tb(matrix_view<> Y, const matrix_view<> A, const matrix_view<> B) noexcept {
    const int N = Y.extent<0>();
    const int M = Y.extent<1>(); 
    const int K = A.extent<1>();

    #pragma omp parallel for if(M * K * N >= 3276800)
    for (int i = 0; i < N; i++) {
        float* __restrict Y_row = Y.data() + i * M;
        const float* __restrict A_row = A.data() + i * K;

        for (int j = 0; j < M; j++) {
            const float* __restrict B_row = B.data() + j * K;
            
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++)
                sum += A_row[k] * B_row[k];

            Y_row[j] += sum;
        }
    }
}

// Y += A^T * B
// Y: [N, M], A: [K, N], B: [K, M]
inline void mat_mul_add_Ta_B(matrix_view<> Y, const matrix_view<> A, const matrix_view<> B) noexcept {
    const int N = A.extent<1>();
    const int M = B.extent<1>();
    const int K = A.extent<0>();

    #pragma omp parallel for if(M * K * N >= 3276800)
    for (int i = 0; i < N; i++) {
        float* __restrict Y_row = Y.data() + i * M;

        for (int k = 0; k < K; k++) {
            const float* __restrict B_row = B.data() + k * M;
            const float a_val = A[k * N + i];

            for (int j = 0; j < M; j++)
                Y_row[j] += a_val * B_row[j];
        }
    }
}

// Y += sum(X)
// Y: [M], X: [N, M]
inline void mat_reduce_sum_axis_0(matrix_view<1> Y, const matrix_view<> X) {
    const int N = X.extent<0>();
    const int M = X.extent<1>();

    for (int i = 0; i < N; i++) {
        const float* __restrict X_row = X.data() + i * M;
        for (int j = 0; j < M; j++)
            Y[j] += X_row[j];
    }
}

}