# RNN.cpp
基于纯 C++ 标准库，无需第三方库依赖的 RNN 训练与推理框架，适合学习与实验用途。

## 如何编译本项目
1. 项目中使用到了部分 C++26 特性，需要支持 **C++26 标准** 的 **C++编译器** (例如 **gcc 16.1.0**, **clang 22.1.2**)

2. Clone 本项目
    ```powershell
    git clone --depth 1 https://github.com/SyrieYume/RNN.cpp
    cd RNN.cpp
    ```

3. 执行下面的编译指令
    ```powershell
    # 如果编译器是clang
    clang++ -std=c++26 rnn_infer.cpp -o rnn_infer.exe -O3 -ffast-math
    clang++ -std=c++26 rnn_train.cpp -o rnn_train.exe -O3 -march=native -ffast-math -fopenmp

    # 如果编译器是gcc
    g++ -std=c++26 rnn_infer.cpp -o rnn_infer.exe -O3 -ffast-math --static
    g++ -std=c++26 rnn_train.cpp -o rnn_train.exe -O3 -march=native -ffast-math -fopenmp --static
    ```

## 模型训练
1. 训练数据必须是 **UTF-8** 编码的文本文件

2. 执行下面的指令
    ```powershell
    ./rnn_train.exe 训练集.txt -o model.bin
    ```

## 模型推理
执行下面的指令
```powershell
./rnn_infer.exe model.bin --temperature 0.85 --top-p 0.9 --max-tokens 128
```


## 注意
Windows 平台使用 **LLVM-MSVC** (LLVM官方Windows版本) 编译生成的程序，在**性能**上可能存在问题，目前在我的测试环境中，性能仅有 gcc 和 LLVM-MinGW 的 30%