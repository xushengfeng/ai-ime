import { assert } from "@std/assert";
import { pinyin } from "pinyin-pro";
import { load_pinyin } from "../key_map/pinyin/gen_zi_pinyin.ts";
import { keys_to_pinyin } from "../key_map/pinyin/keys_to_pinyin.ts";
import { initLIME } from "../main.ts";

const { commit, single_ci, model } = await initLIME({ ziInd: load_pinyin() });

async function test_text_offset(test_text: string[]) {
	let offset = 0;
	const start_time = Date.now();
	for (const _src_t of test_text) {
		let t = "";
		let py = pinyin(_src_t, { toneType: "none" }).replaceAll(" ", "");

		let src_t = _src_t;
		while (py.length > 0) {
			const pinyin_input = keys_to_pinyin(py);
			const candidates = await single_ci(pinyin_input);
			let has = false;

			for (const [idx, candidate] of candidates.candidates.entries()) {
				const text = candidate.word;
				if (src_t.startsWith(text)) {
					has = true;
					src_t = src_t.slice(text.length);
					t = t + text;
					py = candidate.remainkeys.join("");
					console.log(idx, text);
					offset = offset + idx;
					commit(text);
					break;
				}
			}
			if (has === false) {
				console.log("找不到", t);
				break;
			}
		}
	}

	const ttt = Date.now() - start_time;
	console.log("偏移", offset, ttt, ttt / test_text.length);
}

Deno.test("test text offset", async () => {
	const seg = new Intl.Segmenter("zh-Hans", { granularity: "word" });

	const l = Array.from(seg.segment("聪明的输入法")).map((v) => v.segment);
	await test_text_offset(l);
});

Deno.test("test text unnormal", async () => {
	const c = await single_ci(keys_to_pinyin("ku"));
	console.log(
		model.tokenizer("堀"),
		model.tokenizer("堀").map((v) => model.detokenize([v])),
	);
	assert(c.candidates.find((v) => v.word === "堀"));
});
