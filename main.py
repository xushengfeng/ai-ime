from pypinyin import lazy_pinyin
from llama_cpp import Llama
import numpy as np

from typing import List, Dict, Tuple, TypedDict


class Candidate(TypedDict):
    word: str
    score: float
    pinyin: List[str]


BeamList = List[
    Tuple[
        float,  # prob
        str,  # context
        str,  # remaining_pinyin
        List[  # 过程分词
            Tuple[
                str,  # 词
                float,  # 概率
                int,  # 索引
            ]
        ],
        List[str],  # matched_pinyin
    ]
]
model_name = "../Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_0.gguf"
print("加载模型", model_name)

llm = Llama(model_path=model_name, verbose=True, logits_all=True)
print("加载完成")

print("创建拼音索引")

token_pinyin_map: Dict[int, List[str]] = {}

for token_id in range(llm.n_vocab()):
    try:
        token = llm.detokenize([token_id]).decode()
    except:
        continue
    if token:
        token_pinyin_map[token_id] = lazy_pinyin(token)

# 上下文存储
pre_context = "下面的内容主题多样并且没有标点"
user_context = []


# 按键转拼音
def keys_to_pinyin(keys: str) -> str:
    # 示例：将按键直接映射为拼音（实际可根据需求扩展）
    # 比如双拼、模糊
    return keys


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_top_k_logits_numpy(logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (probs, indices)：对选中的 logits 做 softmax 并返回对应的索引（按 logits 降序）。
    :param logits: 一维 numpy 数组
    :param k: 取 top-k 的大小
    :return: (probs, indices)
    """
    logits = np.asarray(logits)
    n = logits.size

    if k >= n:
        sorted_indices = np.argsort(logits)[::-1]
        selected_logits = logits[sorted_indices]
        return softmax(selected_logits), sorted_indices

    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(logits[top_k_indices])[::-1]]
    selected_logits = logits[top_k_indices]
    return softmax(selected_logits), top_k_indices


# 使用 Beam Search 生成候选词，拼音拆分基于候选词
def beam_search_generate(
    pinyin_input: str, max_beam_w=7, min_beam_w=4, top_k: int = 10, pre_str=""
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
            inputs = llm.tokenize(pm.encode())
            print("runmodel", pm)

            llm.reset()
            llm.eval(inputs)

            logits_array = llm._scores[-1]

            logits = np.array(logits_array)

            tk = min(10**5, logits.size)

            top_probs, top_indices = get_top_k_logits_numpy(logits, tk)

            bw = max_beam_w if run_count == 1 else min_beam_w

            for i in range(tk):
                token_prob = top_probs[i]
                if token_prob < 10**-10:
                    break
                new_prob = prob * token_prob  # 累乘概率

                token_id = top_indices[i]
                try:
                    token: str = llm.detokenize([token_id]).decode()
                except:
                    continue
                if len(next_beam) == bw:
                    if new_prob < next_beam[-1][0] and len(token) == 1:
                        break

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
                                ltk + [(token, token_prob, i)],
                                matched_pinyin + token_pinyin,
                                bw,
                            )
                            else 0
                        )

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
    tk: List[Tuple[str, float, int]],
    new_matched_pinyin: List[str],
    limit: int,
):
    """
    将新路径添加到 Beam
    """
    if len(next_beam) == limit:
        if new_prob < next_beam[-1][0] and len(tk[0][0]) == 1:
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
        # 长的优先，然后是prob，但这样的还是有点粗糙
        next_beam.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)
    if len(next_beam) > limit:
        next_beam.pop()
    return True


print("初始化完毕")
