from main import keys_to_pinyin, beam_search_generate, commit
from pypinyin import lazy_pinyin

from typing import List


def test_text_offset(test_text: str):
    """
    测试函数：将提供的文本转为拼音，调用补全引擎，计算文本在候选中的偏移量。

    :param test_text: 测试的输入文本
    """
    print(f"测试文本: {test_text}")

    # 转换为拼音
    pinyin_input = keys_to_pinyin("".join(lazy_pinyin(test_text)))
    print(f"转换为拼音: {pinyin_input}")

    # 调用补全引擎生成候选词
    candidates = beam_search_generate(pinyin_input)
    print("生成的候选词:")
    for idx, candidate in enumerate(candidates):
        print(f"{idx}: {candidate}")

    # 计算偏移量
    offsets = [
        idx
        for idx, candidate in enumerate(candidates)
        if candidate["word"] == test_text
    ]
    if offsets:
        print(f'文本 "{test_text}" 在候选中的偏移量: {offsets[0]}')
    else:
        print(f'文本 "{test_text}" 不在候选中')


if __name__ == "__main__":
    # 示例测试
    commit("测试补全引擎")
    test_text = "测试成功"
    test_text_offset(test_text)
