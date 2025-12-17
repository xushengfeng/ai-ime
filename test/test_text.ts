import { fileURLToPath } from "node:url";
import path from "node:path";
import { pinyin } from "pinyin-pro";
import { keys_to_pinyin } from "../key_map/pinyin/keys_to_pinyin.ts";
import { commit, type Result, single_ci } from "../main.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const file = path.join(__dirname, "冰灯.txt");

const seg = new Intl.Segmenter("zh-Hans", { granularity: "word" });
const test_text_raw = Deno.readTextFileSync(file);
const test_text_x = Array.from(seg.segment(test_text_raw));
const text_text_g: string[][] = [[]];
for (const t of test_text_x) {
	if (!t.isWordLike) {
		text_text_g.push([t.segment]);
		text_text_g.push([]);
		continue;
	}
	if (t.segment.match(/^[\dA-Za-z]+$/)) {
		text_text_g.push([t.segment]);
		text_text_g.push([]);
		continue;
	}
	const last = text_text_g.at(-1);
	if (last && last.length < 5) {
		last.push(t.segment);
	} else {
		text_text_g.push([t.segment]);
	}
}

const test_text = text_text_g.map((v) => v.join(""));
console.log(test_text);

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

let offset = 0;
let keyCount = 0;
const keyT = (1000 * 60) / 120;
const offsetT = 100; // 查找偏移需要的时间

for (let src_t of test_text) {
	let py = pinyin(src_t, { type: "array", toneType: "none", v: true }).join("");
	keyCount += py.length;
	for (let _i = 0; _i < src_t.length; _i++) {
		const c = await single_ci(keys_to_pinyin(py));
		const m = match(src_t, c);
		if (m === undefined) {
			if (!" ，。《》？！“”：/、".includes(src_t)) console.log("找不到", src_t);
			commit(src_t, false, true);
			continue;
		}
		keyCount++; // confirm
		py = m.rm.join("");
		src_t = src_t.slice(m.text.length);
		console.log(m.text, m.idx);
		offset += m.idx;
		if (src_t === "") break;
	}
}

console.log(
	"偏移",
	offset,
	"按键数",
	keyCount,
	"文章长度",
	test_text_raw.length,
	`速度约 ${test_text_raw.length / ((keyCount * keyT + offset * offsetT) / 1000 / 60)} wpm`,
);
