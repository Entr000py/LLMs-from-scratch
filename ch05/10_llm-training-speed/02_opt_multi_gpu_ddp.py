# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import time
import urllib.request

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

# 新增导入（参见附录 A）:
import platform
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# 新增：初始化分布式进程组（每个 GPU 一个进程，用于进程间通信；参见附录 A）
def ddp_setup(rank, world_size):
    """
    参数:
        rank: 该进程在分布式组内的唯一编号
        world_size: 分布式进程总数
    """
    # 如果未由 torchrun 预先设置，则仅在此设置 MASTER_ADDR 与 MASTER_PORT
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # 初始化进程组（进程间通信后端）
    if platform.system() == "Windows":
        # 关闭 libuv；Windows 下的 PyTorch 未内置该支持
        os.environ["USE_LIBUV"] = "0"
        # Windows 上通常需要使用 "gloo"（而非 "nccl"）作为后端
        # gloo：Meta 开源的集体通信库
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl：NVIDIA 集体通信库（Linux 多 GPU 的推荐后端）
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 将当前进程绑定到对应编号的 GPU，以确保 CUDA 操作路由到正确设备
    torch.cuda.set_device(rank)


#####################################
# 第 2 章
#####################################


class GPTDatasetV1(Dataset):
    """
    以滑动窗口方式将长文本切分为重叠序列的数据集。
    
    参数:
        txt: 原始长文本。
        tokenizer: 分词/编码器（如 GPT-2 的 tiktoken 编码器）。
        max_length: 每个样本的最大序列长度（输入 token 数）。
        stride: 窗口步长；小于 max_length 时会产生重叠片段。
    产出:
        __getitem__ 返回 (input_ids, target_ids)，其中 target_ids 为 input_ids 右移一位的预测目标。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对整段文本进行分词/编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口，将文本切分为长度为 max_length 的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 新增：设置 shuffle=False 并使用分布式采样器（参见附录 A）
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, drop_last=True, num_workers=0):
    """
    基于 `GPTDatasetV1` 构建 DataLoader，适配分布式训练的采样方式。

    参数:
        txt: 用于构建数据集的文本。
        batch_size: 批大小。
        max_length: 每个样本的最大序列长度。
        stride: 滑动窗口步长。
        drop_last: 是否丢弃最后一个不足批大小的 batch。
        num_workers: DataLoader 预取线程数。

    说明:
        - shuffle 设为 False 是因为分布式场景下由 `DistributedSampler` 负责打乱与切分样本。
        - `DistributedSampler` 会将样本均匀划分到各个进程/设备，避免重复。
    """
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 构建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 构建 DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # 由于使用 DistributedSampler，此处需设置为 False
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        # 使用分布式采样器，根据 rank/world_size 划分样本，确保各 GPU 样本不重叠
        sampler=DistributedSampler(dataset)
    )
    return dataloader


#####################################
# 第 3 章
#####################################
class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) -> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) -> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) -> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 将 3 个张量分别视为 queries, keys, values，形状均为 (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # 合并多头，其中 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


#####################################
# 第 4 章
#####################################


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = PyTorchMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力模块的残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 张量形状为 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 与残差相加

        # 前馈模块的残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 与残差相加

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # 张量形状为 [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx 的形状为 (B, T)，表示当前上下文的 token 索引
    for _ in range(max_new_tokens):

        # 若超出模型支持的上下文长度，则仅保留最后 context_size 个 token
        idx_cond = idx[:, -context_size:]

        # 前向预测
        with torch.no_grad():
            logits = model(idx_cond)

        # 仅取最后一个时间步；(batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # 取概率最高的下一个 token 索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 形状为 (batch, 1)

        # 将新 token 追加到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)  # 形状变为 (batch, n_tokens+1)

    return idx

#####################################
# 第 5 章
#####################################


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 增加 batch 维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 移除 batch 维度
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, device, start_context):
    """
    使用给定起始上下文在当前设备上生成一段示例文本，并只修改评估用到的状态。

    说明:
        - 若 `model` 被 DDP 包裹，则实际模块位于 `model.module` 中，需据此获取上下文长度。
        - 生成结束后会恢复 `train()` 状态，尽量不干扰训练流程。
    """
    model.eval()

    # 针对 DDP 的处理：若为 DDP 则从 model.module 读取位置嵌入长度
    context_size = model.module.pos_emb.weight.shape[0] if isinstance(model, DDP) else model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tiktoken.get_encoding("gpt2")).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tiktoken.get_encoding("gpt2"))
        print(decoded_text.replace("\n", " "))  # 紧凑打印格式（去除换行）
    model.train()


def train_model_simple_with_timing(model, train_loader, val_loader, optimizer, device,
                                   num_epochs, eval_freq, eval_iter, start_context):
    """
    训练主循环（含计时与分布式吞吐统计）。

    参数:
        model: 训练的模型（单卡或 DDP 包裹）。
        train_loader: 训练数据加载器（分布式下应使用 DistributedSampler）。
        val_loader: 验证数据加载器。
        optimizer: 优化器（如 AdamW）。
        device: 当前训练设备。
        num_epochs: 训练轮数。
        eval_freq: 每多少步触发一次评估与吞吐统计打印。
        eval_iter: 每次评估时，从 dataloader 取多少个 batch 计算平均损失。
        start_context: 用于周期性生成样例文本的起始提示。

    返回:
        train_losses, val_losses, track_tokens：训练/验证损失轨迹与累积 token 数轨迹。
    """
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # NEW: Determine the current rank (default to 0 if not distributed)
    # 当前进程的 rank（未初始化分布式时默认为 0）
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # Variables for cumulative average tokens/sec
    # 用于计算全局平均吞吐的累积量
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # CUDA-specific timing setup
    # CUDA 计时：通过 cuda Event 精确测量 GPU 端耗时；CPU 则使用 time.time
    use_cuda = device.type == "cuda"
    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # 确保之前的 CUDA 操作已完成
        t_start.record()          # 记录本区间起始时间
    else:
        t0 = time.time()          # 记录本区间起始时间（CPU）

    # Main training loop
    for epoch in range(num_epochs):
        # NEW: set epoch for DistributedSampler so each process gets a unique shuffle order
        # 分布式场景需在每个 epoch 设置随机种子以保证各 rank 的采样顺序正确重排
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        for inp_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Forward and backward pass
            loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
            loss.backward()
            optimizer.step()

            # 以 batch 的元素个数（近似 token 数）累计吞吐统计
            total_tokens += inp_batch.numel()

            # At evaluation intervals, measure elapsed time and tokens per second
            if global_step % eval_freq == 0:
                # End timing for the current interval
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()  # 等待本区间所有 CUDA 操作完成
                    elapsed = t_start.elapsed_time(t_end) / 1000  # ms 转秒
                    t_start.record()  # 重置计时器，开始下一区间
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()  # 重置计时器，开始下一区间

                # Calculate local tokens processed during this interval
                # 本区间内本进程处理的 token 数
                local_interval = total_tokens - last_tokens
                last_tokens = total_tokens

                # Aggregate the tokens processed over all devices
                # 通过 all_reduce 求和，得到全局（所有 GPU）在本区间处理的 token 数
                local_tensor = torch.tensor([local_interval], device=device, dtype=torch.float)
                global_tensor = local_tensor.clone()
                torch.distributed.all_reduce(global_tensor, op=torch.distributed.ReduceOp.SUM)
                global_interval = global_tensor.item()

                # Global tokens per second for this interval
                # 区间吞吐（全局）：token / s
                global_tps = global_interval / elapsed if elapsed > 0 else 0

                # Update cumulative tokens (local) and aggregate globally
                # 维护累积量并再次 all_reduce 得到全局累积 token，随后计算全局平均吞吐
                cumulative_tokens += local_interval
                local_cum_tensor = torch.tensor([cumulative_tokens], device=device, dtype=torch.float)
                global_cum_tensor = local_cum_tensor.clone()
                torch.distributed.all_reduce(global_cum_tensor, op=torch.distributed.ReduceOp.SUM)
                global_cumulative_tokens = global_cum_tensor.item()
                cumulative_time += elapsed
                global_avg_tps = global_cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # Evaluate model performance (this may add overhead)
                # 评估当前模型在少量 batch 上的训练/验证损失（会带来额外开销）
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                # NEW: Only print logs once per GPU (choosing the rank 0 GPU)
                # 仅在 rank 0 打印，避免多卡重复输出
                if rank == 0:
                    print(f"Ep {epoch+1}, Step {global_step:06d}, "
                          f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                          f"Step tok/sec: {round(global_tps)}, Global avg tok/sec: {round(global_avg_tps)}")

        # NEW Only rank 0 prints the generated sample and memory usage stats
        if rank == 0 and epoch % 5 == 0:
            generate_and_print_sample(model, device, start_context)

            # Memory stats
            # 打印显存使用情况（仅在 CUDA 可用时）
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # Convert to GB
                reserved = torch.cuda.memory_reserved(current_device) / 1024**3    # Convert to GB

                print(f"\nAllocated memory: {allocated:.4f} GB")
                print(f"Reserved memory: {reserved:.4f} GB\n")

    return train_losses, val_losses, track_tokens


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


#####################################
# 主流程
#####################################

def main(gpt_config, settings, rank, world_size):

    ddp_setup(rank, world_size)  # 初始化分布式进程组
    device = torch.device("cuda", rank)

    torch.manual_seed(123)

    # 仅在 rank 0 输出一次基础环境信息
    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")

            capability = torch.cuda.get_device_capability()
            if capability[0] >= 7:  # Volta/Turing/Ampere/Hopper 及以上架构
                torch.set_float32_matmul_precision("high")
                print("使用 Tensor Cores")
            else:
                print("该 GPU 不支持 Tensor Cores，使用默认精度。")
        print()

    ##############################
    # Download data if necessary
    ##############################

    file_path = "middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    # 仅下载一次数据（由 rank 0 执行）
    if rank == 0:
        if not os.path.exists(file_path):
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode('utf-8')
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)

    # 所有进程等待 rank 0 完成下载（使用当前 GPU 索引同步）
    torch.distributed.barrier(device_ids=[device.index])

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model = torch.compile(model)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    # 使用 DDP 包裹模型
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        fused=True
    )

    ##############################
    # Set up dataloaders
    ##############################

    # 训练/验证集划分比例
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        num_workers=4
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        num_workers=4
    )

    ##############################
    # 开始训练
    ##############################

    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
    )

    # 训练结束，清理分布式进程组
    destroy_process_group()

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # 从环境变量中读取 world_size 与 rank
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    GPT_CONFIG_124M = {
        "vocab_size": 50304,     # 词表大小
        "context_length": 1024,  # 每个训练样本的输入 token 数
        "emb_dim": 768,          # 词向量/隐藏维度
        "n_heads": 12,           # 注意力头数
        "n_layers": 12,          # 层数
        "drop_rate": 0.1,        # Dropout 比例
        "qkv_bias": False        # Q/K/V 线性层是否使用偏置
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,  # 可按 world_size 线性放大（多卡时）
        "num_epochs": 50,
        "batch_size": 32,
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(
        GPT_CONFIG_124M, OTHER_SETTINGS,
        rank, world_size  # NEW
    )

    ###########################
    # 训练后处理
    ###########################

    # 仅由 rank 0 生成一次图表
    if rank == 0:
        # 绘制损失曲线
        epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("loss.pdf")

    # 模型保存/加载（示例，默认注释）
    #
    # compiled = hasattr(model, "_orig_mod")
    # if compiled:
    #     torch.save(model._orig_mod.state_dict(), "model.pth")
    # else:
    #     torch.save(model.state_dict(), "model.pth")
    #
    # model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))
