import { fileURLToPath } from "node:url";
import path from "node:path";
import { pinyin } from "pinyin-pro";
import { keys_to_pinyin } from "../key_map/pinyin/keys_to_pinyin.ts";
import { commit, single_ci } from "../main.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const file = path.join(__dirname, "de.txt");

const test_text = Deno.readTextFileSync(file).split("");
const d: Record<string, { count: number; perfect: number; zis: string[] }> = {
	ta: {
		count: 0,
		perfect: 0,
		zis: ["他", "她", "它"],
	},
	na: {
		count: 0,
		perfect: 0,
		zis: ["那", "哪"],
	},
	de: {
		count: 0,
		perfect: 0,
		zis: ["的", "地", "得"],
	},
	zai: {
		count: 0,
		perfect: 0,
		zis: ["在", "再"],
	},
};

for (const src_t of test_text) {
	const py = pinyin(src_t, { toneType: "none", type: "array" }).join("");
	const pyin = keys_to_pinyin(py);
	const candidates = await single_ci(pyin);
	for (const [i, candidate] of candidates.candidates.entries()) {
		if (src_t === candidate.word) {
			for (const v of Object.values(d)) {
				if (v.zis.includes(src_t)) {
					v.count++;
					if (i === 0) {
						v.perfect++;
					}
					console.log(i, src_t);
				}
			}
			commit(src_t);
			break;
		}
	}
}

console.log(
	d,
	Object.values(d)
		.map((v) => v.perfect)
		.reduce((a, b) => a + b, 0) /
		Object.values(d)
			.map((v) => v.count)
			.reduce((a, b) => a + b, 0),
);
