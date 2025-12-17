# LIME

llm 驱动的输入法。目前支持拼音。

llm 常用的文本生成方式是自回归，也就是预测下一个词（token）的所有可能，然后通过某种方式采样选择某个可能，追加到模型输入，然后再次预测。这个项目，把词可能的选择采样交给了用户拼音，利用用户拼音来辅助采样。

使用小型大模型 Qwen3-0.6B-IQ4_XS ，兼顾速度和联想能力，打字时速度和普通引擎基本无异。

python 版本的见[python 分支](https://github.com/xushengfeng/lime/tree/python)，此版本用 ts 重写。

## 运行

需要有 deno.js 运行时，见[官网](https://deno.com/)

下载本项目，建议通过命令`git clone https://github.com/xushengfeng/lime`，后续可以获取更新，当然也可以下载压缩包

### 安装依赖

```shell
deno install
```

### 下载模型

```shell
git clone https://www.modelscope.cn/unsloth/Qwen3-0.6B-GGUF.git
```

模型文件夹的位置和项目应该是同级的，当然你也可以修改代码

### 开启服务器

```shell
deno serve -A --port 5000 server.ts
```

默认启用自然码双拼，`server.ts`里把`pinyinConfig`的`shuangpin`改成`false`即可关闭

创建密钥，一定程度上防止被滥用或隐私泄露

```shell
deno run -A key.ts
```

## 作为输入法

这里使用[rime](https://rime.im/)作为前端。

复制项目 rime 文件夹里面的内容到你的 rime 输入法配置里面。

安装 luasocket（见下面）。

创建密钥`deno run -A key.ts`，只需要创建一次，把输出的密钥改写在`llm_pinyin.lua`的`key`变量里面。

开启服务器，切换到 llm 拼音输入法即可使用。

注意，并不能与你其他的 rime 输入法结合，只能作为一个新的 rime 输入法。

### 安装 luasocket

对于 linux，使用`luarocks`安装：

```
sudo luarocks install luasocket \
  LIBFLAG="-shared -llua"
```

## 特性

除了 ai 优化，还有一些输入法特性：

-   模糊音
-   自然码双拼
-   `'`号分割拼音

## 现状

不支持长句输入，但输入一长串拼音后，可以中途选择来实现组句。

输入太快可能会漏字母。

没有保存数据的功能，也没有生词记录，所以服务器重启后会丢失记忆。

## 理解与展望

模型有的地方让人惊喜，有些候选又不合适地排在后面。总的来说，联想能力不输以前传统大厂的输入法，利好开源输入法，但 AI 时代竞争会更激烈，大厂的或者新加入的输入法会更智能。

从拼音引导文字生成来看，人对语音的理解不是顺序的，是大体上顺序，小范围逆序，一些音的识别在后面才会有明确的结果或纠正。现在这个项目只能对部分置信度高的候选生成长词组，对于更长的长句，没有一定的把握是不会生成的。我了解到 fim 补全模式，这可以是一个方向，用它来表示没有把握的候选，但并不能提供拼音信息。有几种方向（AI 也告诉我了一些，我不是专业的，仅抛砖引玉），可以修改 mask，让拼音候选匹配的文字权重加大，占用位置编码，但不具体下来；类似翻译模型，前后关系在模型内部处理。为了更好补全，可以微调模型，减少其在指令遵循、编程相关的能力，提高其文学能力。另外发现不同的 token 粒度对置信度影响较大，在“ta de”中，“他的”是一个 token，但“他”和“的”各是一个 token，“他的”排名靠后，但“他”\*“的”的置信度还会更低，所以现在输入法采取长词优先，尽管这个不符合置信度排序。

在应用方面来说，不同焦点的切换应该发生给模型以提示，否则容易串。删除或者光标改变也应该考虑。这些输入法框架应该具有相关功能，我研究一下。

## 开发

可以发送按键让引擎分析

```shell
curl --request POST \
  --url http://127.0.0.1:5000/candidates \
  --header 'content-type: application/json' \
  --header 'Authorization: Bearer your key' \
  --data '{
  "keys": "nihaoshijie"
}'
```

返回

```json
{
    "candidates": [
        {
            "pinyin": ["ni", "hao", "shi", "jie"],
            "score": 1.1879427571978856e-13,
            "word": "你好世界"
        }
    ]
}
```

选好词后，发送，将作为上下文记录

```shell
curl --request POST \
  --url http://127.0.0.1:5000/commit \
  --header 'content-type: application/json' \
  --data '{
  "text": "你好世界"
}'
```
