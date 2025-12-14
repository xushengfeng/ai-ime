import math
from pickletools import pyint
from llama_cpp import Llama
import numpy as np
import threading

from typing import List, Dict, Set, Tuple, TypedDict

from assets.pinyin.script.gen_zi_pinyin import load_pinyin
from utils.keys_to_pinyin import PinyinAndKey, PinyinL


Pinyin = List[str]


class Candidate(TypedDict):
    word: str
    score: float
    pinyin: Pinyin
    remainkeys: List[str]
    preedit: str
    consumedkeys: int


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


class Debounce:
    def __init__(self, delay: float, func):
        self.delay = delay
        self.func = func
        self.timer = None

    def reset(self):
        if self.timer:
            self.timer.cancel()

        self.timer = threading.Timer(self.delay, lambda: self.func())
        self.timer.start()

    def cancel(self):
        if self.timer:
            self.timer.cancel()


model_name = "../Qwen3-0.6B-GGUF/Qwen3-0.6B-IQ4_XS.gguf"
print("加载模型", model_name)

llm = Llama(model_path=model_name, logits_all=True, verbose=False, n_ctx=4096)
print("加载完成")

print("创建拼音索引")

pinyin = load_pinyin()

token_pinyin_map: Dict[int, List[List[str]]] = {}
first_pinyin_token: Dict[str, Set[int]] = {}

for token_id in range(llm.n_vocab()):
    try:
        token = llm.detokenize([token_id]).decode()
    except:
        continue
    if token:
        pys = pinyin(token)
        if pys:
            token_pinyin_map[token_id] = pys
            for fp in pys[0]:
                s = first_pinyin_token[fp] if fp in first_pinyin_token else set()
                s.add(token_id)
                first_pinyin_token[fp] = s


# 上下文存储
pre_context = "下面的内容主题多样"
user_context = []
last_context_data = {"context": ""}

y用户词: Dict[int, Set[int]] = {}

max_count = 4000
rm_count = min(max_count, 64, math.floor(max_count * 0.2))


last_result: np.ndarray | None = None


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

                token_pinyin_dy = token_pinyin_map.get(int(token_id))

                if not (token_pinyin_dy):
                    continue

                check_count += 1
                token_pinyin: List[str] = []
                pyeq = True
                if len(pinyin_input) >= len(token_pinyin_dy):
                    for [i, ps] in enumerate(token_pinyin_dy):
                        input_posi = list(map(lambda x: x["py"], pinyin_input[i]))
                        zi_posi = ps
                        find_zi_eq = False
                        for p in zi_posi:
                            if p in input_posi:
                                find_zi_eq = True
                                token_pinyin.append(p)
                                break
                        if find_zi_eq == False:
                            pyeq = False
                            break
                else:
                    pyeq = False

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
                "consumedkeys": len(
                    "".join(
                        list(
                            map(
                                lambda x: x[0]["key"],
                                pinyin_input[: len(matched_pinyin)],
                            )
                        )
                    )
                ),
            }
        )

    print(run_count, model_count, check_count, add_count)

    # 按得分排序并返回 Top-K
    candidates.sort(key=lambda x: x["score"], reverse=True)

    trim_context.reset()

    return candidates[:top_k]


class li(TypedDict):
    ppy: List[PinyinAndKey]
    tkids: List[int]
    next_ids: Set[int]


def single_ci(pinyin_input: PinyinL) -> Result:
    if not pinyin_input or not pinyin_input[0]:
        return {"candidates": []}

    if last_result is None:
        return {"candidates": []}

    logits = last_result

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

        token_pinyin_dy = token_pinyin_map.get(int(token_id))

        if not (token_pinyin_dy):
            continue
        token_pinyin = pinyin_in_pinyin(pinyin_input, token_pinyin_dy)

        if token_pinyin != False:
            if token != token_pinyin[0]["py"]:
                rmpy = list(
                    map(lambda x: x[0]["key"], pinyin_input[len(token_pinyin) :])
                )

                if rmpy and (token_id in y用户词):
                    # 尝试匹配用户词典
                    lis: List[li] = [
                        li(
                            ppy=token_pinyin.copy(),
                            next_ids=y用户词.get(token_id, set()),
                            tkids=[token_id],
                        )
                    ]
                    final_lis: List[li] = []
                    for _i in range(4):
                        nl: List[li] = []
                        for item in lis:
                            if len(item["next_ids"]) == 0:
                                final_lis.append(item)
                            for i in item["next_ids"]:
                                r = pinyin_input[len(item["ppy"]) :]
                                p = token_pinyin_map.get(i, [])
                                m = pinyin_in_pinyin(r, p)
                                if m:
                                    nitem = li(
                                        ppy=item["ppy"] + m,
                                        next_ids=y用户词.get(i, set()),
                                        tkids=item["tkids"] + [i],
                                    )
                                    nl.append(nitem)
                        lis = nl
                    print(final_lis)
                    for i in final_lis:
                        rmpy1 = list(
                            map(lambda x: x[0]["key"], pinyin_input[len(i["ppy"]) :])
                        )
                        c.append(
                            {
                                "pinyin": list(map(lambda x: x["py"], i["ppy"])),
                                "score": float(token_prob),
                                "word": llm.detokenize(i["tkids"]).decode(),
                                "preedit": " ".join(
                                    list(map(lambda x: x["preeditShow"], i["ppy"]))
                                )
                                + (" " if rmpy1 else ""),
                                "remainkeys": rmpy1,
                                "consumedkeys": len(
                                    "".join(list(map(lambda x: x["key"], i["ppy"])))
                                ),
                            }
                        )

                c.append(
                    {
                        "pinyin": list(map(lambda x: x["py"], token_pinyin)),
                        "score": float(token_prob),
                        "word": token,
                        "remainkeys": rmpy,
                        "preedit": " ".join(
                            map(lambda x: x["preeditShow"], token_pinyin)
                        )
                        + (" " if rmpy else ""),
                        "consumedkeys": len(
                            "".join(list(map(lambda x: x["key"], token_pinyin)))
                        ),
                    }
                )
    c.sort(key=lambda x: len(x["word"]), reverse=True)

    print(
        "token长度",
        llm.n_tokens,
    )

    trim_context.reset()

    if not c:
        print("is empty")
    return {"candidates": c}


def pinyin_in_pinyin(pinyin_input: PinyinL, token_pinyin_dy: List[List[str]]):
    token_pinyin: List[PinyinAndKey] = []
    pyeq = True
    if len(pinyin_input) >= len(token_pinyin_dy):
        for [i, ps] in enumerate(token_pinyin_dy):
            input_posi = pinyin_input[i]
            zi_posi = ps
            find_zi_eq = False
            for p in zi_posi:
                for x in input_posi:
                    if p == x["py"]:
                        find_zi_eq = True
                        token_pinyin.append(x)
                        break
                if find_zi_eq:
                    break
            if find_zi_eq == False:
                pyeq = False
                break
        return token_pinyin if pyeq else False
    else:
        return False


def commit(text: str, update=False, new=True):
    """
    提交

    :param text: 要提交的文本
    :param update: 如果为真，尝试匹配找到追加的字符串
    :param new: 不匹配，输入的就是新的文本
    """
    new_text = ""
    if update:
        if text.startswith(last_context_data["context"]):
            new_text = text[len(last_context_data["context"]) :]
            last_context_data["context"] = text
        else:
            new_text = text
            new = True
    if new:
        last_context_data["context"] = ""
        if update == False:
            new_text = text

    if not new_text:
        return user_context

    # todo 知道最后选了哪个token
    add_user_word(text)

    user_context.append(new_text)

    if llm.n_tokens >= max_count:
        try_trim_context()
        return user_context

    to_run = llm.tokenize(new_text.encode())
    llm.eval(to_run)
    global last_result
    logits_array = llm._scores[-1]
    last_result = np.array(logits_array)

    trim_context.reset()

    return user_context


def get_context():
    return pre_context + "".join(user_context)


def clear_commit():
    user_context.clear()
    llm.reset()
    global last_result
    last_result = None
    init_ctx()


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


def add_user_word(w: str, pre_id: int | None = None):
    ts = llm.tokenize(w.encode())
    if len(ts) > 1:
        for [i, t] in enumerate(ts):
            pre = ts[i - 1] if i != 0 else pre_id
            if pre != None:
                s = y用户词.get(pre, set())
                s.add(t)
                y用户词[pre] = s
        print("添加用户词", w)


def try_trim_context():
    if llm.n_tokens < max_count:
        return
    old_tokens = llm.tokenize("".join(user_context).encode())
    new_tokens = old_tokens[: max_count - rm_count]
    llm.reset()
    for t in new_tokens:
        llm.eval([t])
    print(llm.n_tokens)
    user_context.clear()
    user_context.append(llm.detokenize(new_tokens).decode())
    global last_result
    logits_array = llm._scores[-1]
    last_result = np.array(logits_array)


trim_context = Debounce(10, try_trim_context)


def stop_all():
    trim_context.cancel()


def init_ctx():
    prompt = get_context()
    inputs = llm.tokenize(prompt.encode())
    llm.reset()
    llm.eval(inputs)
    global last_result
    logits_array = llm._scores[-1]
    last_result = np.array(logits_array)


init_ctx()

print("初始化完毕")
