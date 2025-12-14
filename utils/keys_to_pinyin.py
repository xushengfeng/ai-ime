from typing import List, TypedDict
from utils.all_pinyin import generate_pinyin
from utils.fuzzy_pinyin import generate_fuzzy_pinyin
from utils.shuangpin import generate_shuang_pinyin


class PinyinAndKey(TypedDict):
    py: str
    key: str
    isAllMatch: bool


# fmt:off
PinyinL = List[  # 拆分后的序列
    List[PinyinAndKey]  # 多选，比如模糊音，半个拼音等
]
# fmt:on

pinyin_k_l = sorted(
    list(filter(lambda x: len(x) > 1, generate_pinyin())),
    key=lambda x: len(x),
    reverse=True,
)

sp_map = generate_shuang_pinyin(pinyin_k_l)


# 按键转拼音
def keys_to_pinyin(keys: str, shuangpin=True) -> PinyinL:
    # 示例：将按键直接映射为拼音（实际可根据需求扩展）
    # 比如双拼、模糊
    l: PinyinL = []
    k = keys
    if shuangpin != True:
        shuangpin_map = {}
    else:
        shuangpin_map = sp_map

    def try_match(k: str):
        has = False

        for i in shuangpin_map.keys():
            if k.startswith(i):
                has = True
                pinyin = shuangpin_map[i]
                pinyin_variants = generate_fuzzy_pinyin(pinyin)
                py_list: List[PinyinAndKey] = []
                for variant in pinyin_variants:
                    py_list.append(PinyinAndKey(key=i, py=variant, isAllMatch=True))
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
                    py_list.append(PinyinAndKey(key=i, py=variant, isAllMatch=True))
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
                        ll.append(
                            PinyinAndKey(key=xk, py=shuangpin_map[i], isAllMatch=False)
                        )
                for i in pinyin_k_l:
                    if i.startswith(xk):
                        ll.append(PinyinAndKey(key=xk, py=i, isAllMatch=False))
                if ll:
                    l.append(ll)
                k = k[len(xk) :]
                if ll:
                    break
    return l
