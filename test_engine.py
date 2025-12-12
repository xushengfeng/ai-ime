import time
from typing import List
from main import keys_to_pinyin, beam_search_generate, commit, clear_commit, single_ci
from pypinyin import lazy_pinyin
import jieba


def test_text_offset(test_text: List[str]):
    """
    测试函数：将提供的文本转为拼音，调用补全引擎，计算文本在候选中的偏移量。

    :param test_text: 测试的输入文本
    """
    print(f"测试文本: {test_text}")

    offset = 0
    start_time = time.time()
    for src_t in test_text:
        t = ""
        py = "".join(lazy_pinyin(src_t))

        while len(py) > 0:
            pinyin_input = keys_to_pinyin(py)
            candidates = single_ci(pinyin_input, pre_str=t)
            has = False

            for idx, candidate in enumerate(candidates["candidates"]):
                text = candidate["word"]
                if src_t.startswith(text):
                    has = True
                    src_t = src_t[len(text) :]
                    t = t + text
                    py = "".join(candidate["remainkeys"])
                    print(idx, text)
                    offset = offset + idx
                    commit(text)
                    break
            if has == False:
                print("找不到", t)
                break

    ttt = time.time() - start_time
    print("偏移", offset, ttt, ttt / len(test_text))


if __name__ == "__main__":
    # 示例测试
    # commit("测试补全引擎")
    # test_text = "测试成功"
    # test_text_offset(test_text)

    clear_commit()
    test_text_offset(list(jieba.cut("聪明的输入法")))

    clear_commit()
    commit("小明是女的，")
    c = single_ci(keys_to_pinyin("ta"))
    print(c)

    clear_commit()
    test_text_offset(
        list(
            jieba.cut(
                "输入法到一定上下文长度后性能下降似乎是三十二这里进行测试这里需要很长的文字这样够吗要不再多一点确实如此不可接受试试使用缓存或者限制上下文"
            )
        )
    )
    commit("小明是女的，")
    c = single_ci(keys_to_pinyin("ta"))
    print(c)
