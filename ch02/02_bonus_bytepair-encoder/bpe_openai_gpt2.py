# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# License:
# Modified MIT License

# Software Copyright (c) 2019 OpenAI

# We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import regex as re
import requests
from tqdm import tqdm
from functools import lru_cache


@lru_cache()    #缓存
def bytes_to_unicode():
    """
    返回一个UTF-8字节列表和相应的Unicode字符串列表。
    可逆的BPE（Byte Pair Encoding）编码在Unicode字符串上工作。
    这意味着如果你想避免UNK（未知）字符，你的词汇表中需要大量的Unicode字符。
    当处理像100亿个token的数据集时，你需要大约5K个字符才能获得不错的覆盖率。
    这在你通常的32K BPE词汇表中占了相当大的比例。
    为了避免这种情况，我们需要在UTF-8字节和Unicode字符串之间建立查找表。
    并且避免映射到BPE代码无法处理的空白/控制字符。
    """
    # 初始化bs列表，包含ASCII可见字符（!到~）、部分拉丁字符（¡到¬）和扩展拉丁字符（®到ÿ）。
    # 这些字符是GPT-2 BPE编码中直接映射的字节。
    # 33到126，161到172，174到255
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    
    # 初始化cs列表，最初与bs相同。cs将存储映射到的Unicode码点。
    cs = bs[:]
    
    # n用于为未映射的字节生成新的Unicode码点。
    n = 0
    
    # 遍历所有256个可能的字节值（0到255）。
    for b in range(2**8):
        # 如果当前字节b不在bs列表中（即它不是一个直接映射的可见字符），
        # 则将其添加到bs中，并为其分配一个从256开始的新的Unicode码点。
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n) # 将字节映射到256以上的Unicode码点，以避免与现有字符冲突。
            n += 1
            
    # 将cs列表中的整数码点转换为实际的Unicode字符。
    cs = [chr(n) for n in cs]
    
    # 返回一个字典，将原始字节（bs）映射到对应的Unicode字符（cs）。
    # 这个字典用于将字节序列转换为BPE可以处理的Unicode字符串。
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    返回一个单词中所有相邻符号对的集合。
    单词被表示为符号的元组（符号是可变长度的字符串）。
    例如，如果word是("h", "e", "l", "l", "o")，则返回的对将是{("h", "e"), ("e", "l"), ("l", "l"), ("l", "o")}。
    """
    pairs = set() # 初始化一个空集合来存储符号对
    prev_char = word[0] # 将第一个字符设置为前一个字符
    for char in word[1:]: # 遍历单词中的所有字符，从第二个字符开始
        pairs.add((prev_char, char)) # 将当前字符和前一个字符组成的对添加到集合中
        prev_char = char # 更新前一个字符为当前字符，为下一次迭代做准备
    return pairs # 返回所有相邻符号对的集合


class Encoder:
    """
    Encoder类实现了GPT-2的字节对编码（BPE）算法，用于将文本编码为token ID，以及将token ID解码回文本。
    它处理字节到Unicode的映射，并应用一系列合并规则来逐步构建词汇表。
    """
    def __init__(self, encoder, bpe_merges, errors='replace'):
        """
        初始化Encoder实例。

        参数:
            encoder (dict): 从文本token到整数ID的映射字典。
            bpe_merges (list): BPE合并规则的列表，每个规则是一个元组，表示要合并的两个符号。
            errors (str): 解码时如何处理错误（例如，'replace'表示替换无法解码的字节）。
        """
        self.encoder = encoder  # 文本token到整数ID的映射
        self.decoder = {v: k for k, v in self.encoder.items()}  # 整数ID到文本token的反向映射
        self.errors = errors  # 解码错误处理方式
        
        # 获取字节到Unicode字符的映射，用于处理原始字节数据
        self.byte_encoder = bytes_to_unicode()
        # 获取Unicode字符到字节的反向映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 将BPE合并规则转换为一个字典，其中键是合并对，值是它们的优先级（索引）。
        # 优先级越小，合并越早发生。
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}  # 用于缓存BPE处理结果，避免重复计算

        # 用于将文本分割成基本token的正则表达式模式。
        # 这个模式匹配常见的英文单词结构、数字、非字母数字符号和空白。
        # GPT-2的原始实现中提到应该添加re.IGNORECASE以处理大小写变体，但此处未添加。
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        """
        对单个token应用BPE算法，将其分解为子词单元。

        参数:
            token (str): 要进行BPE编码的输入token（已转换为Unicode字符串）。

        返回:
            str: 经过BPE处理后的token，子词单元之间用空格分隔。
        """
        if token in self.cache:
            return self.cache[token] # 如果已缓存，直接返回缓存结果
        
        word = tuple(token) # 将token转换为字符元组，便于处理
        pairs = get_pairs(word) # 获取初始的相邻字符对

        if not pairs:
            return token # 如果没有对，说明token无法再分解，直接返回

        while True:
            # 找到优先级最高的合并对（即在bpe_ranks中索引最小的对）。
            # 如果对不在bpe_ranks中，则优先级视为无穷大。
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break # 如果优先级最高的对不在合并规则中，则停止合并
            
            first, second = bigram # 获取要合并的两个符号
            new_word = [] # 用于构建合并后的新单词
            i = 0
            while i < len(word):
                try:
                    # 查找第一个符号在当前单词中的位置
                    j = word.index(first, i)
                    new_word.extend(word[i:j]) # 将第一个符号之前的部分添加到新单词中
                    i = j
                except ValueError:
                    new_word.extend(word[i:]) # 如果找不到第一个符号，将剩余部分添加到新单词中
                    break

                # 如果当前位置和下一个位置的符号与要合并的bigram匹配
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second) # 合并这两个符号
                    i += 2 # 跳过已合并的两个符号
                else:
                    new_word.append(word[i]) # 如果不匹配，添加当前符号
                    i += 1
            new_word = tuple(new_word) # 更新单词为合并后的元组
            word = new_word
            if len(word) == 1:
                break # 如果单词只剩一个符号，停止合并
            else:
                pairs = get_pairs(word) # 重新计算新的相邻符号对

        word = ' '.join(word) # 将最终的子词单元用空格连接起来
        self.cache[token] = word # 缓存结果
        return word

    def encode(self, text):
        """
        将原始文本编码为BPE token ID列表。

        参数:
            text (str): 要编码的原始文本字符串。

        返回:
            list: 文本对应的整数token ID列表。
        """
        bpe_tokens = []
        # 使用正则表达式模式将文本分割成基本token
        for token in re.findall(self.pat, text):
            # 将每个token的UTF-8字节转换为Unicode字符串，以便BPE处理
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 对转换后的token应用BPE算法，并将其分解为子词单元，然后查找对应的整数ID
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """
        将BPE token ID列表解码回原始文本字符串。

        参数:
            tokens (list): 整数token ID列表。

        返回:
            str: 解码后的原始文本字符串。
        """
        # 将token ID转换回其对应的文本token（Unicode字符串）
        text = ''.join([self.decoder[token] for token in tokens])
        # 将Unicode字符串转换回字节序列，然后使用UTF-8解码回原始文本
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(model_name, models_dir):
    """
    根据给定的模型名称和模型目录，加载GPT-2的编码器和BPE合并规则，并返回一个Encoder实例。

    参数:
        model_name (str): GPT-2模型的名称（例如，"117M"）。
        models_dir (str): 存储模型文件的目录路径。

    返回:
        Encoder: 一个配置好的Encoder实例，用于执行BPE编码和解码。
    """
    # 构建encoder.json文件的完整路径，并加载编码器字典。
    # encoder.json包含从文本到整数token ID的映射。
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    
    # 构建vocab.bpe文件的完整路径，并读取BPE合并规则数据。
    # vocab.bpe包含BPE算法用于合并字节对的规则。
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    
    # 解析bpe_data，将其转换为一个元组列表，每个元组代表一个BPE合并对。
    # 跳过第一行（通常是文件头）和最后一行（通常是空行）。
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    
    # 使用加载的编码器和BPE合并规则创建一个Encoder实例并返回。
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    """
    从OpenAI的公共存储库下载GPT-2模型的词汇表文件（encoder.json和vocab.bpe）。
    这些文件是BPE编码器正常工作所必需的。
    """
    # Modified code from
    subdir = 'gpt2_model' # 定义模型文件存储的子目录名称
    if not os.path.exists(subdir):
        os.makedirs(subdir) # 如果子目录不存在，则创建它
    subdir = subdir.replace('\\', '/')  # needed for Windows # 替换Windows路径分隔符，确保URL路径正确

    # 遍历需要下载的两个文件：encoder.json和vocab.bpe
    for filename in ['encoder.json', 'vocab.bpe']:
        # 构建文件的下载URL，并发送GET请求，使用stream=True以便分块下载
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        # 以二进制写入模式打开文件，用于保存下载内容
        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"]) # 获取文件总大小
            chunk_size = 1000 # 定义每次读取的块大小
            # 使用tqdm显示下载进度条
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                # 迭代响应内容，分块写入文件并更新进度条
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
