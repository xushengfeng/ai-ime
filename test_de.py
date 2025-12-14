# 放在test文件夹后找不到模块，不知道怎么处理……

from os import path
from main import beam_search_generate, commit, clear_commit, single_ci, stop_all
from pypinyin import lazy_pinyin

from utils.keys_to_pinyin import keys_to_pinyin

script_dir = path.dirname(path.abspath(__file__))
file_path = path.normpath(path.join(script_dir, "test", "de.txt"))


if __name__ == "__main__":
    clear_commit()
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    test_text = list(content)

    offset = 0
    count = 0
    perfect = 0
    for src_t in test_text:
        py = "".join(lazy_pinyin(src_t))

        pinyin_input = keys_to_pinyin(py, shuangpin=False)
        candidates = single_ci(pinyin_input)
        has = False

        for idx, candidate in enumerate(candidates["candidates"]):
            text = candidate["word"]
            if src_t == text:
                has = True
                if text in ["他", "她", "它", "那", "哪", "的", "地", "得"]:
                    count = count + 1
                    offset = offset + idx
                    if idx == 0:
                        perfect = perfect + 1
                    print(idx, text)
                commit(text)
                break
        if has == False:
            print("找不到", src_t)
            continue

    print("偏移", offset, "成功", perfect, count)


stop_all()
