from pypinyin import lazy_pinyin
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Tuple, TypedDict


class Candidate(TypedDict):
    word: str
    score: float
    pinyin: List[str]


BeamList = List[Tuple[float, str, str, List[Tuple[str, float]], List[str]]]

# 加载模型和分词器
model_name = "Qwen/Qwen3-0.6B"  # 或您使用的模型
print("加载模型", model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("加载完成")

print("创建拼音索引")


def generate_token_pinyin_map(tokenizer):
    token_pinyin_map: Dict[int, List[str]] = {}
    for token_id in range(tokenizer.vocab_size):
        token = tokenizer.decode([token_id]).strip()
        if token:
            token_pinyin_map[token_id] = lazy_pinyin(token)
    return token_pinyin_map


token_pinyin_map = generate_token_pinyin_map(tokenizer)

# 上下文存储
pre_context = "下面的内容主题多样并且没有标点"
user_context = []


# 按键转拼音
def keys_to_pinyin(keys: str) -> str:
    # 示例：将按键直接映射为拼音（实际可根据需求扩展）
    # 比如双拼、模糊
    return keys


# 使用 Beam Search 生成候选词，拼音拆分基于候选词
def beam_search_generate(
    pinyin_input: str, max_beam_w=7, min_beam_w=2, top_k: int = 10, pre_str=""
) -> List[Candidate]:
    """
    使用 Beam Search 生成候选词，逐步匹配拼音。

    :param pinyin_input: 用户输入的拼音
    :param beam_width: Beam Search 的宽度
    :param top_k: 最终返回的候选词数量
    :return: 候选词列表
    """
    prompt = get_context()

    # 初始化 Beam Search 队列
    beam: BeamList = [
        (1.0, "", pinyin_input, [], [])
    ]  # (prob, context, remaining_pinyin, token_tails, matched_pinyin)

    final_candidates: List[Tuple[float, str, List[str]]] = []

    run_count = 0
    model_count = 0
    check_count = 0
    add_count = 0

    while beam:
        run_count += 1
        print(run_count)
        next_beam: BeamList = []
        for prob, context, remaining_pinyin, ltk, matched_pinyin in beam:
            if not remaining_pinyin:  # 如果拼音已经全部匹配完
                final_candidates.append((prob, context, matched_pinyin))
                continue

            model_count += 1
            pm = prompt + pre_str + context
            inputs = tokenizer(pm, return_tensors="pt")
            print("runmodel", pm)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]

            probabilities = torch.softmax(logits, dim=-1)
            tk = min(10**5, logits.size(-1))
            top_probs, top_indices = torch.topk(probabilities, tk)

            bw = max_beam_w if run_count == 1 else min_beam_w

            for i in range(tk):
                token_prob = top_probs[0, i].item()
                if token_prob < 10**-10:
                    break
                new_prob = prob * token_prob  # 累乘概率

                if len(next_beam) == bw:
                    if new_prob < next_beam[-1][0]:
                        break

                token_id = top_indices[0, i].item()
                token: str = tokenizer.decode([token_id])
                if len(token) < 1:
                    continue
                if token.startswith("\t"):
                    continue
                if token.startswith("\n"):
                    continue
                if token.startswith(" "):
                    continue
                new_context = context + token

                token_pinyin = token_pinyin_map.get(int(token_id))
                if not (token_pinyin):
                    continue
                token_pinyin_str = "".join(token_pinyin)
                check_count += 1
                if remaining_pinyin.startswith(token_pinyin_str):
                    if token != token_pinyin[0]:
                        new_remaining_pinyin = remaining_pinyin[len(token_pinyin_str) :]

                        add_count += (
                            1
                            if add_to_beam(
                                next_beam,
                                new_prob,
                                new_context,
                                new_remaining_pinyin,
                                ltk + [(token, token_prob)],
                                matched_pinyin + token_pinyin,
                                bw,
                            )
                            else 0
                        )

        # 按概率排序并截取 Beam Width 个最优结果
        next_beam.sort(key=lambda x: x[0], reverse=True)  # 按概率从高到低排序
        print(next_beam)
        beam = next_beam

    # 提取最终候选词
    candidates: List[Candidate] = []
    for prob, tokens, matched_pinyin in final_candidates:
        candidates.append({"word": tokens, "score": prob, "pinyin": matched_pinyin})

    print(run_count, model_count, check_count, add_count)

    # 按得分排序并返回 Top-K
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def commit(text: str):
    user_context.append(text)


def get_context():
    return pre_context + "".join(user_context)


def clear_commit():
    user_context.clear()


def add_to_beam(
    next_beam: BeamList,
    new_prob: float,
    new_context: str,
    new_remaining_pinyin: str,
    tk: List[Tuple[str, float]],
    new_matched_pinyin: List[str],
    limit: int,
):
    """
    将新路径添加到 Beam
    """
    if len(next_beam) == limit:
        if new_prob < next_beam[-1][0]:
            return False

    print(new_prob, new_context, new_remaining_pinyin, tk, new_matched_pinyin)
    next_beam.append(
        (
            new_prob,
            new_context,
            new_remaining_pinyin,
            tk,
            new_matched_pinyin,
        )
    )
    if len(next_beam) >= limit:
        next_beam.sort(key=lambda x: x[0], reverse=True)
    if len(next_beam) > limit:
        next_beam.pop()
    return True


print("初始化完毕")
