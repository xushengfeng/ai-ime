from functools import wraps
from key import verify_key
from main import keys_to_pinyin, beam_search_generate, commit, clear_commit, single_ci

from flask import Flask, request, jsonify, abort
from typing import List, Dict
from urllib.parse import unquote

# 初始化 Flask 应用
print("初始化网络服务器")
app = Flask(__name__)

# 添加一个全局变量存储用户上下文（假设这是你需要的）
user_context = []

auth = True


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        b = request.headers.get("Authorization", "")
        if auth:
            if b.startswith("Bearer "):
                token = b[len("Bearer ") :]
                if not verify_key(token):
                    abort(403)
            else:
                abort(401)
        return f(*args, **kwargs)

    return decorated


# API: 获取候选词 - POST 方法
@app.route("/candidates", methods=["POST"])
@require_auth
def get_candidates_post() -> Dict[str, List[Dict[str, float]]]:
    data = request.json
    keys: str = data.get("keys", "")  # type: ignore
    pre_str: str = data.get("pre_str", "")  # type: ignore

    pinyin_input = keys_to_pinyin(keys)
    result = single_ci(pinyin_input, pre_str=pre_str)

    return jsonify(result)  # type: ignore


# API: 获取候选词 - GET 方法
@app.route("/candidates", methods=["GET"])
@require_auth
def get_candidates_get() -> Dict[str, List[Dict[str, float]]]:
    # 从 URL 参数获取数据
    keys: str = request.args.get("keys", "")
    pre_str: str = request.args.get("pre_str", "")

    # URL 解码参数
    keys = unquote(keys)
    pre_str = unquote(pre_str)

    pinyin_input = keys_to_pinyin(keys)
    result = single_ci(pinyin_input, pre_str=pre_str)

    return jsonify(result)  # type: ignore


# API: 获取长句 - POST 方法
@app.route("/sentence", methods=["POST"])
@require_auth
def get_sentence_post() -> Dict[str, List[Dict[str, float]]]:
    data = request.json
    keys: str = data.get("keys", "")  # type: ignore
    pre_str: str = data.get("pre_str", "")  # type: ignore

    pinyin_input = keys_to_pinyin(keys)
    candidates = beam_search_generate(pinyin_input, pre_str=pre_str)

    return jsonify({"candidates": candidates})  # type: ignore


# API: 获取长句 - GET 方法
@app.route("/sentence", methods=["GET"])
@require_auth
def get_sentence_get() -> Dict[str, List[Dict[str, float]]]:
    # 从 URL 参数获取数据
    keys: str = request.args.get("keys", "")
    pre_str: str = request.args.get("pre_str", "")

    # URL 解码参数
    keys = unquote(keys)
    pre_str = unquote(pre_str)

    pinyin_input = keys_to_pinyin(keys)
    candidates = beam_search_generate(pinyin_input, pre_str=pre_str)

    return jsonify({"candidates": candidates})  # type: ignore


# API: 提交文字 - POST 方法
@app.route("/commit", methods=["POST"])
@require_auth
def commit_text_post() -> Dict[str, List[str]]:
    data = request.json
    text = data.get("text", "")  # type: ignore

    if not text:
        return jsonify({"error": "No text provided"}), 400  # type: ignore

    commit(text)
    # 更新用户上下文
    user_context.append(text)
    print("".join(user_context)[-20:])

    return jsonify(
        {
            "message": "Text committed successfully",
        }
    )  # type: ignore


# API: 提交文字 - GET 方法
@app.route("/commit", methods=["GET"])
@require_auth
def commit_text_get() -> Dict[str, List[str]]:
    # 从 URL 参数获取数据
    text: str = request.args.get("text", "")
    text = unquote(text)

    if not text:
        return jsonify({"error": "No text provided"}), 400  # type: ignore

    commit(text)
    # 更新用户上下文
    user_context.append(text)
    print("".join(user_context)[-20:])

    return jsonify(
        {
            "message": "Text committed successfully",
        }
    )  # type: ignore


# API: 清除上下文 - GET 方法
@app.route("/clear", methods=["GET"])
@require_auth
def clear_context_get() -> Dict[str, str]:
    global user_context
    user_context = []
    clear_commit()
    return jsonify({"message": "Context cleared successfully"})  # type: ignore


# API: 清除上下文 - POST 方法
@app.route("/clear", methods=["POST"])
@require_auth
def clear_context_post() -> Dict[str, str]:
    global user_context
    user_context = []
    clear_commit()
    return jsonify({"message": "Context cleared successfully"})  # type: ignore


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
