# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

import time
import math
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, max_seq_len=None, window_size=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 预计算缩放常量，避免每次前向开方与除法
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # NEW
        # 设置最大序列长度，如果未提供则使用上下文长度
        self.max_seq_len = max_seq_len or context_length
        # 设置窗口大小，如果未提供则使用最大序列长度
        self.window_size = window_size or self.max_seq_len
        # 注册一个非持久性缓冲区用于存储键缓存
        self.register_buffer("cache_k", None, persistent=False)
        # 注册一个非持久性缓冲区用于存储值缓存
        self.register_buffer("cache_v", None, persistent=False)
        # 环形缓冲区指针与有效长度
        self.ptr_cur = 0
        self.cache_len = 0
        ####################################################

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        queries = queries.transpose(1, 2)

        ####################################################
        # NEW
        # 如果使用缓存
        if use_cache:
            # 初始化缓存（或当batch大小变化时重新初始化）
            if self.cache_k is None or self.cache_k.size(0) != b:
                self.cache_k = torch.empty(
                    b, self.num_heads, self.window_size, self.head_dim,
                    device=x.device, dtype=keys_new.dtype
                )
                self.cache_v = torch.empty_like(self.cache_k)
                self.ptr_cur = 0
                self.cache_len = 0

            # 仅保留当前块中最后 window_size 个token
            tokens_to_write = min(num_tokens, self.window_size)
            ksrc = keys_new[:, :, -tokens_to_write:, :]
            vsrc = values_new[:, :, -tokens_to_write:, :]

            # 分两段写入环形缓冲区，避免整体搬移
            first_part = min(self.window_size - self.ptr_cur, tokens_to_write)
            if first_part > 0:
                self.cache_k[:, :, self.ptr_cur:self.ptr_cur + first_part, :] = ksrc[:, :, :first_part, :]
                self.cache_v[:, :, self.ptr_cur:self.ptr_cur + first_part, :] = vsrc[:, :, :first_part, :]
            second_part = tokens_to_write - first_part
            if second_part > 0:
                self.cache_k[:, :, 0:second_part, :] = ksrc[:, :, first_part:, :]
                self.cache_v[:, :, 0:second_part, :] = vsrc[:, :, first_part:, :]

            # 更新指针与有效长度
            self.ptr_cur = (self.ptr_cur + tokens_to_write) % self.window_size
            self.cache_len = min(self.window_size, self.cache_len + tokens_to_write)

            # 读取按时间顺序排列的键值（必要时拼接两段）
            K_valid = self.cache_len
            if K_valid == 0:
                keys = self.cache_k[:, :, :0, :]
                values = self.cache_v[:, :, :0, :]
            else:
                start = (self.ptr_cur - K_valid) % self.window_size
                if start + K_valid <= self.window_size:
                    keys = self.cache_k[:, :, start:start + K_valid, :]
                    values = self.cache_v[:, :, start:start + K_valid, :]
                else:
                    part1 = self.window_size - start
                    keys = torch.cat([
                        self.cache_k[:, :, start:, :],
                        self.cache_k[:, :, :K_valid - part1, :]
                    ], dim=2)
                    values = torch.cat([
                        self.cache_v[:, :, start:, :],
                        self.cache_v[:, :, :K_valid - part1, :]
                    ], dim=2)
        else:
            # 不使用缓存，直接使用新的键和值
            keys, values = keys_new, values_new
            self.ptr_cur = 0  # 如果交错模式，保持指针正常
            self.cache_len = 0
        ####################################################
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = (queries * self.scale) @ keys.transpose(2, 3)  # 预缩放 queries

        ####################################################
        # NEW
        K = attn_scores.size(-1)

        # 如果当前令牌数量等于K（即没有使用缓存或缓存已满）
        if num_tokens == K:
            causal_mask = torch.triu(
                torch.ones((num_tokens, K), device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        else:
            # 已缓存：通过 (K - num_tokens) 偏移对角线
            offset = K - num_tokens
            causal_mask = torch.triu(
                torch.ones((num_tokens, K), device=x.device, dtype=torch.bool),
                diagonal=1 + offset,
            )
        ####################################################

        # Use the mask to fill attention scores
        neg_inf = torch.finfo(attn_scores.dtype).min
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), neg_inf)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        if self.training:
            attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    ####################################################
    # NEW
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_cur = 0
        self.cache_len = 0
    ####################################################


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.LayerNorm):
    def __init__(self, emb_dim):
        super().__init__(normalized_shape=emb_dim, eps=1e-5)


class GELU(nn.GELU):
    def __init__(self):
        super().__init__(approximate='tanh')


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            window_size=cfg["kv_window_size"] if "kv_window_size" in cfg else cfg["context_length"]   # NEW
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        # NEW
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        # NEW
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.ptr_current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        # NEW

        # 如果使用缓存
        if use_cache:
            # 计算位置ID，从当前指针位置开始
            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            # 更新当前指针位置
            self.ptr_current_pos += seq_len
        else:
            # 不使用缓存，从0开始计算位置ID
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        # 获取位置嵌入并增加一个维度
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # NEW
        # 遍历所有Transformer块并应用
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    ####################################################
    # NEW
    # 重置KV缓存
    def reset_kv_cache(self):
        # 遍历所有Transformer块并重置其注意力模块的缓存
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        # 重置当前位置指针
        self.ptr_current_pos = 0
    ####################################################


# 简单的文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx 是当前上下文中的索引数组 (B, T)
    for _ in range(max_new_tokens):

        # 如果当前上下文超过支持的上下文大小，则裁剪
        # 例如，如果LLM只支持5个令牌，上下文大小为10
        # 那么只有最后5个令牌用作上下文
        idx_cond = idx[:, -context_size:]

        # 获取预测
        with torch.no_grad(): # 在不计算梯度的模式下运行
            logits = model(idx_cond)

        # 只关注最后一个时间步
        # (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        logits = logits[:, -1, :]

        # 获取具有最高logit值的词汇表条目的索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样索引附加到正在运行的序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


####################################################
# NEW
# 使用KV缓存生成文本的简化函数
def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    model.eval() # 将模型设置为评估模式

    # 获取上下文长度，如果未提供则使用模型的位置嵌入数量
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad(): # 在不计算梯度的模式下运行
        if use_cache: # 如果使用缓存
            model.reset_kv_cache() # 重置KV缓存
            # 对初始输入进行模型前向传播，并使用缓存
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens): # 循环生成新的令牌
                # 获取下一个令牌的索引（具有最高logit值的令牌）
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 将新生成的令牌添加到序列中
                idx = torch.cat([idx, next_idx], dim=1)
                # 对新生成的令牌进行模型前向传播，并使用缓存
                logits = model(next_idx, use_cache=True)
        else: # 如果不使用缓存
            for _ in range(max_new_tokens): # 循环生成新的令牌
                # 对当前上下文进行模型前向传播，不使用缓存
                logits = model(idx[:, -ctx_len:], use_cache=False)
                # 获取下一个令牌的索引
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 将新生成的令牌添加到序列中
                idx = torch.cat([idx, next_idx], dim=1)

    return idx # 返回生成的令牌序列
####################################################


# 主函数
def main():
    # GPT模型配置
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "emb_dim": 768,          # 嵌入维度
        "n_heads": 12,           # 注意力头数量
        "n_layers": 12,          # 层数
        "drop_rate": 0.1,        # Dropout 比率
        "qkv_bias": False,       # Query-Key-Value 偏置
        "kv_window_size": 1024   # 新增：KV 缓存窗口大小
    }

    torch.manual_seed(123) # 设置随机种子以保证可复现性
    model = GPTModel(GPT_CONFIG_124M) # 实例化GPT模型
    # 根据CUDA可用性设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # 将模型移动到指定设备
    model.eval()  # 禁用dropout，将模型设置为评估模式

    start_context = "Hello, I am" # 初始文本上下文

    tokenizer = tiktoken.get_encoding("gpt2") # 获取GPT2编码器
    encoded = tokenizer.encode(start_context) # 编码初始文本
    # 将编码后的文本转换为张量，并增加一个批次维度
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize() # 如果CUDA可用，同步CUDA操作
    start = time.time() # 记录开始时间

    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=encoded_tensor,
    #     max_new_tokens=200,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )

    ####################################################
    # NEW
    # 使用带缓存的文本生成函数
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )
    ####################################################

    if torch.cuda.is_available():
        torch.cuda.synchronize() # 如果CUDA可用，同步CUDA操作
    total_time = time.time() - start # 计算总耗时

    # 解码生成的令牌ID为文本
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated() # 获取最大内存分配
        max_mem_gb = max_mem_bytes / (1024 ** 3) # 转换为GB
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main() # 调用主函数
