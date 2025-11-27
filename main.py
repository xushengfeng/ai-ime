from flask import Flask, request, jsonify
from pypinyin import lazy_pinyin
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
from heapq import heappush, heappop

# 初始化 Flask 应用
app = Flask(__name__)

# 加载模型和分词器
model_name = "Qwen/Qwen3-0.6B"  # 或您使用的模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 上下文存储
user_context = [" "]


# 按键转拼音
def keys_to_pinyin(keys: str) -> str:
    # 示例：将按键直接映射为拼音（实际可根据需求扩展）
    # 比如双拼、模糊
    return keys


# 根据拼音筛选候选词
def filter_candidates_by_pinyin(
    candidates: List[Dict[str, float]], pinyin_input: str
) -> List[Dict[str, float]]:
    filtered_candidates = []
    for candidate in candidates:
        word = candidate["word"]
        word_pinyin = "".join(lazy_pinyin(word))
        if pinyin_input in word_pinyin:
            filtered_candidates.append(candidate)
    return filtered_candidates


# 使用 Beam Search 生成候选词，拼音拆分基于候选词
def beam_search_generate(
    pinyin_input: str, beam_width: int = 8, top_k: int = 10
) -> List[Dict[str, float]]:
    """
    使用 Beam Search 生成候选词，逐步匹配拼音。

    :param pinyin_input: 用户输入的拼音
    :param beam_width: Beam Search 的宽度
    :param top_k: 最终返回的候选词数量
    :return: 候选词列表
    """
    prompt = "".join(user_context)
    inputs = tokenizer(prompt, return_tensors="pt")

    # 初始化 Beam Search 队列
    beam = [
        (1.0, "", [], pinyin_input, ("", 1.0))
    ]  # (prob, context, tokens, remaining_pinyin, (token_tail, token_tail_prob))

    final_candidates = []

    while beam:
        next_beam = []
        for prob, context, tokens, remaining_pinyin, (
            token_tail,
            token_tail_prob,
        ) in beam:
            if not remaining_pinyin:  # 如果拼音已经全部匹配完
                final_candidates.append((prob, tokens))
                continue

            print(context, token_tail, prob)
            if token_tail:  # 如果有未处理的 token_tail
                token = token_tail[0]  # 取出 token_tail 的第一个字
                token_tail = token_tail[1:]  # 更新 token_tail
                new_prob = token_tail_prob  # 使用 token_tail 的概率
                new_context = context + token
                new_tokens = tokens + [token]

                # 检查拼音匹配
                token_pinyin = lazy_pinyin(token)
                if remaining_pinyin.startswith(token_pinyin[0]):
                    print(
                        token,
                        new_prob,
                        (
                            new_prob,
                            new_context,
                            new_tokens,
                            new_remaining_pinyin,
                            (token_tail, token_tail_prob),
                        ),
                    )
                    new_remaining_pinyin = remaining_pinyin[len(token_pinyin[0]) :]
                    heappush(
                        next_beam,
                        (
                            new_prob,
                            new_context,
                            new_tokens,
                            new_remaining_pinyin,
                            (token_tail, token_tail_prob),
                        ),
                    )
                continue

            inputs = tokenizer(prompt + context, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]

            probabilities = torch.softmax(logits, dim=-1)
            tk = 1000
            top_probs, top_indices = torch.topk(probabilities, tk)

            for i in range(tk):
                token_id = top_indices[0, i].item()
                token = tokenizer.decode([token_id])
                token_prob = top_probs[0, i].item()
                new_prob = prob * token_prob  # 累乘概率
                new_context = context + token[0]
                new_tokens = tokens + [token[0]]
                new_token_tail = token[1:]  # 提取 token 的剩余部分

                # 检查拼音匹配
                token_pinyin = lazy_pinyin(token[0])
                if remaining_pinyin.startswith(token_pinyin[0]):
                    print(token, token_prob)
                    new_remaining_pinyin = remaining_pinyin[len(token_pinyin[0]) :]
                    heappush(
                        next_beam,
                        (
                            new_prob,
                            new_context,
                            new_tokens,
                            new_remaining_pinyin,
                            (new_token_tail, token_prob),
                        ),
                    )

        # 按概率排序并截取 Beam Width 个最优结果
        next_beam.sort(key=lambda x: x[0], reverse=True)  # 按概率从高到低排序
        beam = next_beam[:beam_width]

    # 提取最终候选词
    candidates = []
    for prob, tokens in final_candidates:
        candidates.append({"word": "".join(tokens), "score": prob})

    # 按得分排序并返回 Top-K
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def commit(text: str):
    user_context.append(text)


# API: 获取候选词
@app.route("/candidates", methods=["POST"])
def get_candidates() -> Dict[str, List[Dict[str, float]]]:
    data = request.json
    keys: str = data.get("keys", "")

    pinyin_input = keys_to_pinyin(keys)
    candidates = beam_search_generate(pinyin_input)

    return jsonify({"candidates": candidates})


# API: 提交文字
@app.route("/commit", methods=["POST"])
def commit_text() -> Dict[str, List[str]]:
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    commit(text)

    return jsonify({"message": "Text committed successfully", "context": user_context})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
