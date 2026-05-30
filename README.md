# RNN.cpp
基于纯 C++ 标准库的 RNN 训练与推理的实现

## 如何编译本项目
1. 需要支持 **C++26标准** 的 **C++编译器** (例如 **gcc 16.1.0**)

2. Clone 本项目:
    ```powershell
    git clone --depth 1 https://github.com/SyrieYume/RNN.cpp
    cd RNN.cpp
    ```

3. 执行下面的编译指令 (以gcc编译器为例):
    ```powershell
    g++ -std=c++26 rnn_infer.cpp -o rnn_infer.exe -O3 -ffast-math --static
    g++ -std=c++26 rnn_train.cpp -o rnn_train.exe -O3 -march=native -ffast-math -fopenmp --static
    ```

## 模型训练
执行下面的指令:
```powershell
./rnn_train.exe 训练集.txt -o model.bin
```

## 模型推理
执行下面的指令:
```powershell
./rnn_infer.exe model.bin --temperature 0.85 --top-p 0.9 --max-tokens 128
```