import time
from pypinyin import lazy_pinyin
from llama_cpp import Llama
import numpy as np

from typing import List, Dict, Set, Tuple, TypedDict

from utils.shuangpin import generate_shuang_pinyin
from utils.split_pinyin import spilt_pinyin


class PinyinAndKey(TypedDict):
    py: str
    key: str


PinyinL = List[  # 拆分后的序列
    List[PinyinAndKey]  # 多选，比如模糊音，半个拼音等
]
Pinyin = List[str]


class Candidate(TypedDict):
    word: str
    score: float
    pinyin: Pinyin
    remainkeys: List[str]
    preedit: str


class Result(TypedDict):
    candidates: List[Candidate]


BeamList = List[
    Tuple[
        float,  # prob
        str,  # context
        PinyinL,  # remaining_pinyin
        List[  # 过程分词
            Tuple[
                str,  # 词
                float,  # 概率
                int,  # 索引
            ]
        ],
        Pinyin,  # matched_pinyin
    ]
]
model_name = "../Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_0.gguf"
print("加载模型", model_name)

llm = Llama(model_path=model_name, logits_all=True, verbose=False)
print("加载完成")

print("创建拼音索引")

token_pinyin_map: Dict[int, List[str]] = {}
first_pinyin_token: Dict[str, Set[int]] = {}
pinyin_k: Set[str] = set()

for token_id in range(llm.n_vocab()):
    try:
        token = llm.detokenize([token_id]).decode()
    except:
        continue
    if token:
        py = lazy_pinyin(token)
        token_pinyin_map[token_id] = py

        fp = py[0]
        s = first_pinyin_token[fp] if fp in first_pinyin_token else set()
        s.add(token_id)
        first_pinyin_token[fp] = s

        py2 = lazy_pinyin(token, errors="ignore")
        for i in py2:
            pinyin_k.add(i)

pinyin_k_l = sorted(
    list(filter(lambda x: len(x) > 1, pinyin_k)), key=lambda x: len(x), reverse=True
) + ["a", "o", "e"]

# 上下文存储
pre_context = "下面的内容主题多样"
user_context = []

to_run: List[int] = []


class FuzzyConfig:
    def __init__(self):
        # 声母模糊音
        self.initial_fuzzy = {
            "c": "ch",
            "z": "zh",
            "s": "sh",
            # 'l': 'n',
            # 'n': 'l',
            # 'f': 'h',
            # 'h': 'f',
            # 'r': 'l',
            # 'l': 'r',
        }

        # 韵母模糊音
        self.final_fuzzy = {
            "an": "ang",
            "ang": "an",
            "en": "eng",
            "eng": "en",
            "in": "ing",
            "ing": "in",
            "ian": "iang",
            "iang": "ian",
            "uan": "uang",
            "uang": "uan",
        }

        # 是否启用模糊音
        self.enabled = True


# 创建全局模糊音配置实例
fuzzy_config = FuzzyConfig()


def generate_fuzzy_pinyin(pinyin: str) -> List[str]:
    """生成模糊音变体"""
    fuzzy_variants = set()

    if not fuzzy_config.enabled:
        return [pinyin]

    initial, final = spilt_pinyin(pinyin)

    for i in [initial] + (
        [fuzzy_config.initial_fuzzy[initial]]
        if initial in fuzzy_config.initial_fuzzy
        else []
    ):
        for j in [final] + (
            [fuzzy_config.final_fuzzy[final]]
            if final in fuzzy_config.final_fuzzy
            else []
        ):
            fuzzy_variants.add(i + j)

    return list(fuzzy_variants)


shuangpin_map = generate_shuang_pinyin(pinyin_k_l)


# 按键转拼音
def keys_to_pinyin(keys: str) -> PinyinL:
    # 示例：将按键直接映射为拼音（实际可根据需求扩展）
    # 比如双拼、模糊
    l: PinyinL = []
    k = keys

    def try_match(k: str):
        has = False

        for i in shuangpin_map.keys():
            if k.startswith(i):
                has = True
                pinyin = shuangpin_map[i]
                pinyin_variants = generate_fuzzy_pinyin(pinyin)
                py_list: List[PinyinAndKey] = []
                for variant in pinyin_variants:
                    py_list.append(PinyinAndKey(key=i, py=variant))
                l.append(py_list)
                k = k[len(i) :]
                return k

        for i in pinyin_k_l:
            if k.startswith(i):
                has = True
                pinyin = i
                pinyin_variants = generate_fuzzy_pinyin(pinyin)
                py_list: List[PinyinAndKey] = []
                for variant in pinyin_variants:
                    py_list.append(PinyinAndKey(key=i, py=variant))
                l.append(py_list)
                k = k[len(i) :]
                return k
        if has == False:
            return None

    count = 0
    while len(k) > 0:
        count = count + 1
        if count > len(keys) * 2:
            break
        nk = try_match(k)
        if nk != None:
            k = nk
        else:
            for plen in range(len(k)):
                xk = k[0 : plen + 1]
                ll: List[PinyinAndKey] = []
                for i in shuangpin_map.keys():
                    if i.startswith(xk):
                        ll.append(PinyinAndKey(key=xk, py=shuangpin_map[i]))
                for i in pinyin_k_l:
                    if i.startswith(xk):
                        ll.append(PinyinAndKey(key=xk, py=i))
                if ll:
                    l.append(ll)
                k = k[len(xk) :]
                if ll:
                    break
    print(l)
    return l


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
        return selected_logits, sorted_indices

    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(logits[top_k_indices])[::-1]]
    selected_logits = logits[top_k_indices]
    return selected_logits, top_k_indices


# 使用 Beam Search 生成候选词，拼音拆分基于候选词
def beam_search_generate(
    pinyin_input: PinyinL, max_beam_w=7, min_beam_w=4, top_k: int = 10, pre_str=""
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

            ftokenid: Set[int] = set()
            for firstPinyin in remaining_pinyin[0]:
                s = first_pinyin_token.get(firstPinyin["py"]) or set()
                ftokenid = ftokenid | s

            for i in range(top_indices.size):
                token_prob = top_probs[i]
                if token_prob < 10**-10:
                    break
                new_prob = prob * token_prob  # 累乘概率

                token_id = top_indices[i]
                if not (token_id in ftokenid):
                    continue
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

                check_count += 1
                pyeq = True
                for [_i, p] in enumerate(token_pinyin):
                    if len(remaining_pinyin) <= _i:
                        pyeq = False
                        break
                    if p not in map(lambda x: x["py"], remaining_pinyin[_i]):
                        pyeq = False
                        break
                if pyeq:
                    if token != token_pinyin[0]:
                        new_remaining_pinyin = remaining_pinyin[len(token_pinyin) :]

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
        candidates.append(
            {
                "word": tokens,
                "score": float(prob),
                "pinyin": matched_pinyin,
                "remainkeys": [],
                "preedit": " ".join(matched_pinyin),
            }
        )

    print(run_count, model_count, check_count, add_count)

    # 按得分排序并返回 Top-K
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def single_ci(pinyin_input: PinyinL, pre_str="") -> Result:
    if not pinyin_input or not pinyin_input[0]:
        return {"candidates": []}
    pm = pre_str

    devTime = time.time()
    inputs = to_run + llm.tokenize(pm.encode())
    devTimeTk = time.time()
    llm.eval(inputs)
    devTimeRun = time.time()

    logits_array = llm._scores[-1]

    logits = np.array(logits_array)

    tk = logits.size

    top_probs, top_indices = get_top_k_logits_numpy(logits, tk)

    ftokenid: Set[int] = set()
    for firstPinyin in pinyin_input[0]:
        s = first_pinyin_token.get(firstPinyin["py"]) or set()
        ftokenid = ftokenid | s

    c: List[Candidate] = []

    for i in range(top_indices.size):
        token_prob = top_probs[i]

        token_id = top_indices[i]
        if not (token_id in ftokenid):
            continue
        try:
            token: str = llm.detokenize([token_id]).decode()
        except:
            continue

        if len(token) < 1:
            continue
        if token.startswith("\t"):
            continue
        if token.startswith("\n"):
            continue
        if token.startswith(" "):
            continue

        token_pinyin = token_pinyin_map.get(int(token_id))
        if not (token_pinyin):
            continue

        pyeq = True
        for [_i, p] in enumerate(token_pinyin):
            if len(pinyin_input) <= _i:
                pyeq = False
                break
            if p not in map(lambda x: x["py"], pinyin_input[_i]):
                pyeq = False
                break
        if pyeq:
            if token != token_pinyin[0]:
                rmpy = list(
                    map(lambda x: x[0]["key"], pinyin_input[len(token_pinyin) :])
                )
                c.append(
                    {
                        "pinyin": token_pinyin,
                        "score": float(token_prob),
                        "word": token,
                        "remainkeys": rmpy,
                        "preedit": pre_str + " ".join(token_pinyin + rmpy),
                    }
                )
    c.sort(key=lambda x: len(x["word"]), reverse=True)

    devTimePy = time.time()

    print(
        "token长度",
        llm.n_tokens,
        "token耗时",
        devTimeTk - devTime,
        "运行耗时",
        devTimeRun - devTimeTk,
        "检索",
        devTimePy - devTimeRun,
    )

    if not c:
        print("is empty")
    return {"candidates": c}


def commit(text: str):
    user_context.append(text)
    global to_run
    to_run = llm.tokenize(text.encode())
    global llmState
    llmState = llm.save_state()
    return user_context


def get_context():
    return pre_context + "".join(user_context)


def clear_commit():
    user_context.clear()
    llm.reset()
    to_run.clear()
    global llmState
    llmState = init_ctx()


def add_to_beam(
    next_beam: BeamList,
    new_prob: float,
    new_context: str,
    new_remaining_pinyin: PinyinL,
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


def init_ctx():
    prompt = get_context()
    inputs = llm.tokenize(prompt.encode())
    global to_run
    to_run = inputs
    llm.reset()
    llm.eval(inputs)
    return llm.save_state()


llmState = init_ctx()

print("初始化完毕")
