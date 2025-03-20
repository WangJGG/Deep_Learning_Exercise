**诗歌生成模型报告**

## 1. RNN、LSTM 和 GRU 模型介绍

循环神经网络（RNN）是一种适用于处理序列数据的神经网络结构。其核心思想是维护一个隐藏状态（hidden state），用于存储历史信息，并通过递归方式更新。

### **1.1 RNN（循环神经网络）**

![image-20250320165755322](F:\Deep_Learning_Exercise\chap6_RNN\report_img\RNN.png)

RNN 通过一个循环结构处理序列数据，每个时间步的隐藏状态由前一个时间步的隐藏状态和当前输入决定。其数学表达式如下：

$ h_t = f(W_h h_{t-1} + W_x x_t + b) $
$ y_t = V h_t $

其中：
- $h_t$ 表示当前时间步的隐藏状态，
- $x_t$ 是当前时间步的输入，
- $W_h, W_x, b, V$ 是模型的参数，
- $f$ 通常为非线性激活函数（如 tanh 或 ReLU）。

RNN 存在梯度消失问题，使得长期依赖信息难以学习。

### **1.2 LSTM（长短时记忆网络）**

![image-20250320165852446](F:\Deep_Learning_Exercise\chap6_RNN\report_img\LSTM.png)

LSTM 通过引入门控机制（遗忘门、输入门、输出门）解决了 RNN 的梯度消失问题。

LSTM 的核心计算如下：

- 遗忘门：$ f_t = \sigma(W_f [h_{t-1}, x_t]^T + b_f) $
- 输入门：$ i_t = \sigma(W_i [h_{t-1}, x_t]^T + b_i) $
- 输出门：$ o_t = \sigma(W_o [h_{t-1}, x_t]^T + b_o) $
- 候选记忆：$ \tilde{C}_t = \tanh(W_C [h_{t-1}, x_t]^T + b_C) $
- 记忆单元更新：$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $
- 隐藏状态更新：$ h_t = o_t * \tanh(C_t) $

LSTM 适用于长序列的建模，能有效捕捉长期依赖关系。

### **1.3 GRU（门控循环单元）**

![image-20250320170003995](F:\Deep_Learning_Exercise\chap6_RNN\report_img\GRU.png)

GRU 是 LSTM 的简化版本，它合并了遗忘门和输入门，使计算更加高效。

GRU 计算方式如下：
- 更新门：$ z_t = \sigma(W_z [h_{t-1}, x_t]^T + b_z) $
- 重置门：$ r_t = \sigma(W_r [h_{t-1}, x_t]^T + b_r) $
- 候选隐藏状态：$ \tilde{h}_t = \tanh(W_h [r_t * h_{t-1}, x_t]^T + b_h) $
- 计算当前隐藏状态：$ h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t $

GRU 计算量更小，适用于高效建模。

---

## **2. 诗歌生成过程**

本项目的诗歌生成基于 RNN 模型，主要包括以下几个步骤：

### 2.1 TensorFlow版本：

#### **2.1.1 数据处理**

1. 读取 `poems.txt` 诗歌数据，每行为一首诗。
2. 为每首诗添加特殊起始符 `bos` 和结束符 `eos`，以便模型学习文本的边界。
3. 统计所有字符的出现频率，建立词汇表，将每个字符映射为索引，构建 `word2id`（字符到索引的映射）和 `id2word`（索引到字符的映射）。
4. 使用 `TensorFlow Dataset` 进行批处理，数据会被填充（padding）到相同长度。

#### **2.1.2 模型构建**

模型 `myRNNModel` 结构如下：

- **词嵌入层（Embedding）**：将字符索引转换为向量，向量维度为 64。
- **RNN 层（SimpleRNN）**：包含 128 个单元，处理序列数据，输出隐藏状态。
- **全连接层（Dense）**：输出词汇表大小的预测概率。

#### **2.1.3 训练过程**

1. **计算损失**：
   - 使用 `sparse_softmax_cross_entropy_with_logits` 计算交叉熵损失。
   - 目标序列为输入序列的后移版本（即输入 `x` 生成目标 `y`）。
2. **优化器**：
   - 采用 `Adam` 优化器，学习率设为 0.0005。
   - 通过 `GradientTape` 计算梯度并更新模型参数。
3. **训练**：
   - 每个批次进行前向传播计算损失，共训练10个批次。
   - 反向传播计算梯度，更新模型权重。
   - 每隔 500 步打印一次损失值。

#### **2.1.4 诗歌生成**

1. **输入起始词（如“日”）**：
   - 使用 `word2id` 将起始字符转换为索引。
   - 设定 RNN 的初始隐藏状态。
2. **逐步预测下一个字符**：
   - 将前一个预测结果作为下一个时间步的输入。
   - 通过 `argmax` 选取概率最高的词。
   - 直到生成结束符 `eos` 或达到最大长度。
3. **转换为文本输出**：
   - 使用 `id2word` 将索引转换回字符。
   - 拼接成完整诗句。

### 2.2 PyTorch版本

#### **2.2.1 数据处理**

1. **数据读取**
   - 诗歌数据存储于 `poems.txt` 文件中，每行为一首诗。
   - 读取诗歌数据时，为每首诗添加特殊起始符 `start_token='G'` 和结束符 `end_token='E'`，确保模型能够识别文本边界。
   - 过滤特殊字符，如 `_ ( ) 《 》 [ ]`，去除过短（少于 5 字）或过长（超过 80 字）的诗歌。
2. **构建词汇表**
   - 统计所有字符的出现频率，按照词频排序。
   - 构建 `word2id`（字符到索引的映射）和 `id2word`（索引到字符的映射），用于将文本转换为数字序列。
   - 诗歌数据转换为索引列表 `poems_vector`，用于训练。
3. **生成批量数据**
   - 训练时使用 **batch_size=64**。
   - `generate_batch(batch_size, poems_vector, word_to_int)` 方法用于切分数据，每批次数据划分为 `x`（输入序列）和 `y`（目标序列）。
   - 目标序列 `y` 是输入序列 `x` 向右偏移一个时间步后的版本。

#### **2.2.2 模型构建**

本项目的 PyTorch 模型 `RNN_model` 主要包含以下结构：

1. **词嵌入层（Embedding）**：
   - `word_embedding` 类用于将字符索引转换为嵌入向量。
   - 嵌入维度设为 `embedding_dim=100`。
2. **LSTM 层**：
   - `RNN_model` 采用 **2 层 LSTM**，隐藏层维度 `lstm_hidden_dim=128`。
   - 设定 `batch_first=True`，确保输入形状为 `(batch_size, seq_length, embedding_dim)`。
   - 初始化 **隐藏状态** `**h0**` **和 细胞状态** `**c0**` **为 0**。
3. **全连接层（Linear）**：
   - `fc` 负责将 LSTM 输出转换为词汇表大小的概率分布。
   - 使用 **ReLU 激活函数** 进行非线性变换。
   - 最终输出使用 **LogSoftmax** 计算类别概率。

#### **2.2.3 训练过程**

1. **损失函数**

   - 采用 **NLLLoss（负对数似然损失）**，适用于 LogSoftmax 输出。

2. **优化器**

   - 使用 **RMSprop** 进行梯度更新，学习率设为 `lr=0.01`。
   - 训练时使用 `clip_grad_norm` 限制梯度范数，避免梯度爆炸。

3. **训练循环（30 轮）**

   - 读取 `poems_vector` 并生成 `batches_inputs, batches_outputs`。
   - 遍历 `batch_size=100` 的数据，进行前向传播计算损失。
   - 反向传播计算梯度，更新模型权重。
   - 每 20 批次保存一次模型。

   **训练过程：**
   ![image-20250320170946524](F:\Deep_Learning_Exercise\chap6_RNN\report_img\torch_training_1.png)

   ![image-20250320171041442](F:\Deep_Learning_Exercise\chap6_RNN\report_img\torch_training_2.png)

#### **2.2.4 诗歌生成**

1. **输入起始词（如“日”）**
   - `gen_poem(begin_word)` 以 `begin_word` 作为起始字符。
   - 将字符转换为索引，并输入到 LSTM 模型。
2. **逐步预测下一个字符**
   - 模型基于当前输入预测下一个字符的概率分布。
   - 采用 **argmax 选择概率最高的词**，并将其添加到诗歌序列中。
   - 直到生成结束符 `E` 或达到最大长度（30 字）。
3. **转换为文本输出**
   - `to_word(predict, vocabs)` 将索引转换回字符。
   - `pretty_print_poem(poem)` 负责格式化诗歌，按 **整句**（逗号、句号）进行换行。
4. **示例**
   - 调用 `gen_poem()` 生成以 **日、红、山、夜、湖、海、月** 开头的诗句。

---

## 3. 生成诗歌示例

使用 `日、红、山、夜、湖、海、月` 作为开头，生成诗歌如下：

### 3.1 TensorFlow版本

![image-20250320170435674](F:\Deep_Learning_Exercise\chap6_RNN\report_img\tf_results.png)

### 3.2 PyTorch版本

![image-20250320171305810](F:\Deep_Learning_Exercise\chap6_RNN\report_img\torch_result.png)

---



