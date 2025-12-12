import time
from main import keys_to_pinyin, beam_search_generate, commit, clear_commit, single_ci
from pypinyin import lazy_pinyin


def test_text_offset(test_text: str):
    """
    测试函数：将提供的文本转为拼音，调用补全引擎，计算文本在候选中的偏移量。

    :param test_text: 测试的输入文本
    """
    print(f"测试文本: {test_text}")

    offset = 0
    src_t = test_text
    t = ""
    py = "".join(lazy_pinyin(src_t))

    start_time = time.time()

    while len(py) > 0:
        pinyin_input = keys_to_pinyin(py)
        candidates = single_ci(pinyin_input, pre_str=t)
        has = False

        for idx, candidate in enumerate(candidates["candidates"]):
            if src_t.startswith(candidate["word"]):
                has = True
                src_t = src_t[len(candidate["word"]) :]
                t = t + candidate["word"]
                py = "".join(candidate["remainkeys"])
                print(idx, candidate["word"])
                offset = offset + idx
                break
        if has == False:
            print("找不到", t)
            break

    ttt = time.time() - start_time
    print(ttt, ttt / len(test_text))


if __name__ == "__main__":
    # 示例测试
    # commit("测试补全引擎")
    # test_text = "测试成功"
    # test_text_offset(test_text)

    clear_commit()
    test_text_offset("聪明的输入法")
