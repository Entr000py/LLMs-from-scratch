"""
Direct Preference Optimization (DPO) Implementation
"""
import json
import os
import urllib.request
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial
from pathlib import Path
import shutil

# 1. 数据集处理
class PreferenceDataset(Dataset):
    """偏好数据集类，用于DPO训练"""
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            # 格式化输入提示（prompt）
            prompt = format_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            # 对prompt进行分词编码
            prompt_tokens = tokenizer.encode(prompt)

            # 构建包含prompt和chosen响应的完整文本，并编码
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            chosen_full_tokens = tokenizer.encode(chosen_full_text)

            # 构建包含prompt和rejected响应的完整文本，并编码
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            # 保存编码后的结果
            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

    def __getitem__(self, index):
        # 返回指定索引的数据条目的编码结果
        return self.encoded_texts[index]

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

def format_input(entry):
    """
    按照Alpaca风格格式化输入提示。
    
    参数:
        entry (dict): 包含 'instruction' 和可选 'input' 键的字典。
        
    返回:
        str: 格式化后的提示字符串，包含指令和输入（如果有）。
    """
    # 构建指令部分文本
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 如果有输入内容，则添加输入部分文本
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    # 返回拼接后的完整输入文本
    return instruction_text + input_text

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    """
    批量整理函数，用于将样本列表转换为模型可用的张量，并进行填充和掩码处理。

    参数:
        batch (list): 一个批次的数据，每个元素为字典，包含 'prompt', 'chosen', 'rejected' 等键。
        pad_token_id (int): 填充用的token id，默认为50256。
        allowed_max_length (int或None): 允许的最大序列长度，超出则截断。
        mask_prompt_tokens (bool): 是否对prompt部分进行掩码处理。
        device (str或torch.device): 张量存放的设备。

    返回:
        dict: 包含填充后的 'chosen', 'rejected' 及其掩码的字典。
    """
    import torch
    from typing import Dict, List, Union
    
    # 初始化批量数据字典
    batch_data: Dict[str, List[Union[List[int], torch.Tensor]]] = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []
    }

    # 计算批次中最长序列长度，用于统一填充
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # 遍历每个样本，处理填充和掩码
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded), dtype=torch.bool)

            # 填充部分掩码为False
            mask[len(sequence):] = False

            # prompt部分掩码为False（+2表示"### Response"前的两个换行符）
            if mask_prompt_tokens:
                mask[:prompt.shape[0] + 2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # 堆叠并移动到指定设备
    tensor_batch_data: Dict[str, torch.Tensor] = {}
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        tensor_stack = torch.stack(batch_data[key])  # type: ignore
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]
        tensor_batch_data[key] = tensor_stack.to(device)
    
    # 处理prompt数据
    tensor_batch_data["prompt"] = torch.stack(batch_data["prompt"]).to(device)  # type: ignore
    
    return tensor_batch_data

def download_and_load_file(file_path, url):
    """下载并加载JSON文件"""
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

# 2. 模型定义和加载
def load_model_and_tokenizer(CHOOSE_MODEL="gpt2-medium (355M)"):
    """加载模型和分词器"""
    from previous_chapters import GPTModel
    
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    
    # 加载模型
    finetuned_model_path = Path("gpt2-medium355M-sft.pth")
    if not finetuned_model_path.exists():
        # Try finding the model checkpoint locally:
        relative_path = Path("..") / "01_main-chapter-code" / finetuned_model_path
        if relative_path.exists():
            shutil.copy(relative_path, ".")
        else:
            raise FileNotFoundError(
                f"Could not find '{finetuned_model_path}'.\n"
                "Run the `ch07.ipynb` notebook to finetune and save the finetuned model."
            )
    
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(
        torch.load(
            "gpt2-medium355M-sft.pth",
            map_location=torch.device("cpu"),
            weights_only=True
        )
    )
    model.eval()
    
    # 加载分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    
    return model, tokenizer, BASE_CONFIG

# 3. DPO损失函数
def compute_logprobs(logits, labels, selection_mask=None):
    """
    计算每个样本的平均对数概率。

    参数:
      logits: 张量，形状为 (batch_size, num_tokens, vocab_size)，模型输出的logits。
      labels: 张量，形状为 (batch_size, num_tokens)，真实标签（token id）。
      selection_mask: 可选张量，形状为 (batch_size, num_tokens)，用于选择有效token（如去除padding）。

    返回:
      avg_log_prob: 每个样本的平均对数概率（去除padding），形状为 (batch_size,)
    """

    # 标签右移一位，对齐预测
    labels = labels[:, 1:].clone()

    # logits也右移一位，保持与labels一致
    logits = logits[:, :-1, :]

    # 计算每个token的对数概率
    log_probs = F.log_softmax(logits, dim=-1)

    # 选取labels对应的对数概率
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        # 掩码也右移一位
        mask = selection_mask[:, 1:].clone()

        # 只保留有效token的对数概率
        selected_log_probs = selected_log_probs * mask

        # 计算每个样本的平均对数概率（去除padding）
        mask_sum = mask.sum(-1)
        # 避免除零错误
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        avg_log_prob = selected_log_probs.sum(-1) / mask_sum

        return avg_log_prob
    else:
        # 未提供掩码时，直接对所有token取均值
        return selected_log_probs.mean(-1)

def compute_dpo_loss(
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
    beta=0.1,
    ):
    """
    计算一批次策略模型和参考模型的DPO损失。

    参数:
      model_chosen_logprobs: 策略模型对被选响应的对数概率，形状：(batch_size,)
      model_rejected_logprobs: 策略模型对被拒响应的对数概率，形状：(batch_size,)
      reference_chosen_logprobs: 参考模型对被选响应的对数概率，形状：(batch_size,)
      reference_rejected_logprobs: 参考模型对被拒响应的对数概率，形状：(batch_size,)
      beta: DPO损失的温度参数，通常在0.1到0.5之间。beta趋近于0时忽略参考模型。

    返回:
      三个张量组成的元组：(loss, chosen_rewards, rejected_rewards)
    """

    # 策略模型和参考模型的对数几率差
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # DPO损失公式（参考论文Eq. 7）
    losses = -F.logsigmoid(beta * logits)

    # 可选：用于训练过程中的奖励追踪
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # 返回批次平均损失和奖励
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    """
    在一个批次上计算DPO损失。

    参数:
      batch: 数据批次，包含 'chosen', 'rejected' 及其掩码等字段。
      policy_model: 策略模型（待优化的模型）。
      reference_model: 参考模型（保持不变）。
      beta: DPO损失的温度超参数。

    返回:
      损失值、chosen奖励、rejected奖励
    """

    # 策略模型对被选响应的对数概率
    policy_chosen_log_probas = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    # 策略模型对被拒响应的对数概率
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )
    
    # 参考模型对被选和被拒响应的对数概率（不参与梯度计算）
    with torch.no_grad():
        ref_chosen_log_probas = compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        ref_rejected_log_probas = compute_logprobs(
            logits=reference_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )
    # 计算DPO损失
    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )
    return loss, chosen_rewards, rejected_rewards

def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):
    """
    对整个数据加载器应用 compute_dpo_loss_batch，计算平均DPO损失和奖励。

    参数:
        data_loader: 数据加载器（DataLoader），批量提供数据。
        policy_model: 策略模型（待优化的模型）。
        reference_model: 参考模型（保持不变）。
        beta: DPO损失的温度超参数。
        num_batches: 可选，指定计算的批次数，默认为全部批次。

    返回:
        平均损失、平均chosen奖励、平均rejected奖励
    """

    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    if len(data_loader) == 0:
        return float("nan"), float("nan"), float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果指定的批次数超过数据加载器实际批次数，则取实际批次数
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()
        else:
            break

    # 计算平均值
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards

def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """
    计算训练集和验证集的DPO损失。

    参数:
        policy_model: 策略模型（待优化的模型）。
        reference_model: 参考模型（保持不变）。
        train_loader: 训练集数据加载器。
        val_loader: 验证集数据加载器。
        beta: DPO损失的温度超参数。
        eval_iter: 用于评估的批次数。

    返回:
        包含训练和验证损失及奖励的字典。
    """
    policy_model.eval()  # 设置为评估模式
    with torch.no_grad():
        # 计算训练集DPO损失和奖励
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_dpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

        # 计算验证集DPO损失和奖励
        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_rewards,
        "train_rejected_reward": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards
    }

    policy_model.train()  # 恢复为训练模式
    return res

# 4. 训练循环
def train_model_dpo_simple(
    policy_model, reference_model, train_loader, val_loader,
    optimizer, num_epochs, beta,
    eval_freq, eval_iter, start_context, tokenizer
):
    """训练DPO模型的主循环"""
    from previous_chapters import generate_and_print_sample
    
    # 初始化用于跟踪损失和已处理token数量的列表
    tracking = {
        "train_losses": [],           # 训练集损失
        "train_chosen_rewards": [],   # 训练集chosen奖励
        "train_rejected_rewards": [], # 训练集rejected奖励
        "val_losses": [],             # 验证集损失
        "val_chosen_rewards": [],     # 验证集chosen奖励
        "val_rejected_rewards": [],   # 验证集rejected奖励
        "tokens_seen": []             # 已处理token数量
    }
    tokens_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        policy_model.train()  # 设置模型为训练模式

        for batch in train_loader:
            optimizer.zero_grad()  # 清除上一批次的梯度

            # 计算DPO损失和奖励
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )

            loss.backward()      # 反向传播计算梯度
            optimizer.step()     # 更新模型参数

            tokens_seen += batch["chosen"].numel()  # 累加已处理token数量
            global_step += 1

            # 可选：每隔eval_freq步进行一次评估
            if global_step % eval_freq == 0:
                res = evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )

        # 每10个epoch打印一次生成示例
        if (epoch + 1) % 10 == 0:
            generate_and_print_sample(
                model=policy_model,
                tokenizer=tokenizer,
                device=loss.device,
                start_context=start_context
            )

    return tracking

# 5. 主函数
def main():
    """主函数，演示DPO训练流程"""
    # 加载数据
    file_path = "instruction-data-with-preference.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/04_preference-tuning-with-dpo/instruction-data-with-preference.json"
    )
    
    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))
    
    # 划分数据集
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    
    # 加载模型和分词器
    policy_model, tokenizer, BASE_CONFIG = load_model_and_tokenizer()
    
    # 创建参考模型
    from previous_chapters import GPTModel
    reference_model = GPTModel(BASE_CONFIG)
    reference_model.load_state_dict(
        torch.load(
            "gpt2-medium355M-sft.pth",
            map_location=torch.device("cpu"),
            weights_only=True
        )
    )
    reference_model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model.to(device)
    reference_model.to(device)
    
    # 创建数据加载器
    customized_collate_fn = partial(
        custom_collate_fn,
        device=str(device),       # Put the data directly on a GPU if available
        mask_prompt_tokens=True,  # This is optional
        allowed_max_length=1024   # The supported context length of the model
    )
    
    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = PreferenceDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = PreferenceDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = PreferenceDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    # 初始化优化器并开始训练
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)
    
    num_epochs = 1
    tracking = train_model_dpo_simple(
        policy_model=policy_model,
        reference_model=reference_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        beta=0.1, # value between 0.1 and 0.5
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[2]),
        tokenizer=tokenizer
    )
    
    return tracking

if __name__ == "__main__":
    tracking = main()