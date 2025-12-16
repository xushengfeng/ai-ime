import { fileURLToPath } from "node:url";
import path from "node:path";
import { pinyin } from "pinyin-pro";
import { keys_to_pinyin } from "../key_map/pinyin/keys_to_pinyin.ts";
import { commit, type Result, single_ci } from "../main.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const file = path.join(__dirname, "冰灯.txt");

const seg = new Intl.Segmenter("zh-Hans", { granularity: "word" });
const test_text = Array.from(seg.segment(Deno.readTextFileSync(file))).map(
	(i) => i.segment,
);

let offset = 0;

function match(src_t: string, r: Result) {
	for (const [idx, candidate] of r.candidates.entries()) {
		const text = candidate.word;
		if (src_t.startsWith(text)) {
			if (src_t === text) {
				commit(text, true, true);
			} else {
				commit(text, true, false);
			}
			return { text, idx, rm: candidate.remainkeys };
		}
	}
}

let count = 0;
for (let src_t of test_text) {
	count++;
	let py = pinyin(src_t, { type: "array", toneType: "none" }).join("");
	for (let _i = 0; _i < src_t.length; _i++) {
		const c = await single_ci(keys_to_pinyin(py));
		const m = match(src_t, c);
		if (m === undefined) {
			console.log("找不到", src_t);
			commit(src_t, false, true);
			continue;
		}
		py = m.rm.join("");
		src_t = src_t.slice(m.text.length);
		console.log(m.text, m.idx);
		offset += m.idx;
		if (src_t === "") break;
	}
}

console.log("偏移", offset, "分词数", count);
