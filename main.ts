import { fileURLToPath } from "node:url";
import path from "node:path";
import { getLlama, type Token } from "node-llama-cpp";
import { load_pinyin } from "./key_map/pinyin/gen_zi_pinyin.ts";
import type { PinyinAndKey, PinyinL } from "./key_map/pinyin/keys_to_pinyin.ts";
import { pinyin_in_pinyin } from "./utils/pinyin_in_pinyin.ts";

type Candidate = {
	word: string;
	score: number;
	pinyin: string[];
	remainkeys: string[];
	preedit: string;
	consumedkeys: number;
};

export type Result = {
	candidates: Candidate[];
};

type UserData = {
	words: Record<number, Array<Array<number>>>;
	context: Array<string>;
};

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const token_pinyin_map: Map<number, Array<Array<string>>> = new Map();
const first_pinyin_token = new Map<string, Set<number>>();

const pre_context = "下面的内容主题多样";
const user_context: string[] = [];
const last_context_data = { context: "" };
const y用户词 = new Map<number, Array<Array<number>>>();

const max_count = 4000;
const rm_count = Math.min(max_count, 64, Math.floor(max_count * 0.2));

let last_result: Map<Token, number> | undefined;

export function commit(text: string, update = false, newT = true) {
	let new_text = "";
	let nt = newT;

	if (update) {
		if (text.startsWith(last_context_data.context)) {
			new_text = text.slice(last_context_data.context.length);
			last_context_data.context = text;
		} else {
			new_text = text;
			nt = true;
		}
	}
	if (nt) {
		last_context_data.context = "";
		if (update === false) {
			new_text = text;
		}
	}
	if (!new_text) return;

	if (newT) add_user_word(text);

	user_context.push(new_text);

	// todo shift context

	const to_run = model.tokenizer(new_text);
	if (to_run.length === 0) return;

	const pre = to_run.slice(0, -1);
	const last = to_run[to_run.length - 1];
	const release = modelEvalLock.lock();
	sequence
		.controlledEvaluate([
			...pre,
			[
				last,
				{
					generateNext: {
						probabilities: true,
					},
				},
			],
		])
		.then((res) => {
			last_result = res.at(-1)?.next.probabilities;
			release();
		});

	// todo trim context reset
}

function add_user_word(w: string) {
	const ts = model.tokenizer(w);
	if (ts.length === 0) return false;
	const l = y用户词.get(ts[0]) ?? [];
	l.push(ts);
	y用户词.set(ts[0], l);
	return true;
}

export async function single_ci(pinyin_input: PinyinL): Promise<Result> {
	if (pinyin_input.length === 0 || pinyin_input[0].length === 0) {
		return { candidates: [] };
	}

	if (!last_result) {
		return { candidates: [] };
	}

	const ftokenid = new Set<number>();
	for (const firstPinyin of pinyin_input[0]) {
		const s = first_pinyin_token.get(firstPinyin.py) ?? new Set();
		for (const tokenid of s) ftokenid.add(tokenid);
	}

	const c: Candidate[] = [];

	await modelEvalLock.acquire();

	for (const [token_id, token_prob] of last_result) {
		if (!ftokenid.has(token_id)) continue;
		const token = model.detokenize([token_id]);
		if (!token) continue;
		if (["\t", "\n", " "].includes(token[0])) continue;

		const token_pinyin_dy = token_pinyin_map.get(token_id);

		if (!token_pinyin_dy) continue;

		const token_pinyin = pinyin_in_pinyin(pinyin_input, token_pinyin_dy);
		if (!token_pinyin) continue;
		if (token === token_pinyin[0].py) continue; // 排除部分英文

		const rmpy = pinyin_input.slice(token_pinyin.length).map((v) => v[0].py);

		if (rmpy.length > 0 && y用户词.has(token_id)) {
			// todo 用户词
			type li = {
				ppy: PinyinAndKey[];
				tkids: Token[];
				remainids: Token[];
			};
			let lis: li[] = [];
			for (const n of y用户词.get(token_id) || []) {
				lis.push({
					ppy: structuredClone(token_pinyin),
					tkids: [token_id],
					remainids: n.slice(1) as Token[],
				});
			}
			const final_lis: li[] = [];
			for (let _i = 0; _i < 4; _i++) {
				const nl: li[] = [];
				for (const item of lis) {
					const i = item.remainids[0];
					const r = pinyin_input.slice(item.ppy.length);
					if (r.length === 0) break;
					const p = token_pinyin_map.get(i) || [];
					const m = pinyin_in_pinyin(r, p);
					if (m) {
						const rids = item.remainids.slice(1);
						const nitem: li = {
							ppy: item.ppy.concat(m),
							remainids: rids,
							tkids: item.tkids.concat(i),
						};
						if (rids.length === 0) {
							final_lis.push(nitem);
						} else {
							nl.push(nitem);
						}
					}
				}
				lis = nl;
			}
			for (const i of final_lis) {
				const rmpy = pinyin_input.slice(i.ppy.length).map((i) => i[0].key);
				c.push({
					pinyin: i.ppy.map((i) => i.py),
					score: token_prob,
					word: model.detokenize(i.tkids),
					preedit:
						i.ppy.map((i) => i.preeditShow).join(" ") +
						(rmpy.length ? " " : ""),
					remainkeys: rmpy,
					consumedkeys: i.ppy.map((i) => i.key).join("").length,
				});
			}
		}

		c.push({
			pinyin: token_pinyin.map((v) => v.py),
			score: token_prob,
			word: token,
			remainkeys: rmpy,
			preedit:
				token_pinyin.map((v) => v.preeditShow).join(" ") +
				(rmpy.length ? " " : ""),
			consumedkeys: token_pinyin.map((v) => v.key).join("").length,
		});
	}

	c.sort((a, b) => b.pinyin.length - a.pinyin.length);

	console.log("token长度", sequence.contextTokens.length);

	// todo trim_context.reset()

	if (c.length === 0) {
		console.log("is empty");
	}
	return { candidates: c };
}

function get_context() {
	return pre_context + user_context.join("");
}

async function init_ctx() {
	const prompt = get_context();
	const tokens = model.tokenizer(prompt);
	const x = await sequence.controlledEvaluate([
		...tokens.slice(0, -1),
		[
			tokens.at(-1)!,
			{
				generateNext: {
					probabilities: true,
					options: {
						topK: Infinity,
					},
				},
			},
		],
	]);
	last_result = x.at(-1)?.next.probabilities;
}

export function getUserData(): UserData {
	return {
		words: Object.fromEntries(y用户词),
		context: user_context,
	};
}

export function loadUserData(data: UserData) {
	if (y用户词.size > 0 || user_context.length) {
		console.log("已存在用户数据");
		return;
	}
	user_context.length = 0;
	for (const i of data.context) user_context.push(i);
	y用户词.clear();
	for (const [k, v] of Object.entries(data.words)) y用户词.set(Number(k), v);
}

class Lock {
	pm: Promise<void> | null = null;

	async acquire() {
		if (this.pm) await this.pm;
	}

	lock() {
		const p = Promise.withResolvers<void>();
		this.pm = p.promise;
		return p.resolve;
	}
}

const modelEvalLock = new Lock();

const llama = await getLlama({
	gpu: false,
});

const modelPath = "../Qwen3-0.6B-GGUF/Qwen3-0.6B-IQ4_XS.gguf";

console.log("加载模型", modelPath);

const model = await llama.loadModel({
	modelPath: path.join(__dirname, modelPath),
});
const context = await model.createContext({
	contextSize: { max: 4096 },
});
const sequence = context.getSequence();

console.log("加载完成");

console.log("创建拼音索引");

const pinyin = load_pinyin();

for (const token_id of model.iterateAllTokens()) {
	const token = model.detokenize([token_id]);
	if (!token) continue;
	const pinyins = pinyin(token);
	if (pinyins.length) {
		token_pinyin_map.set(token_id, pinyins);
		for (const fp of pinyins[0]) {
			const s = first_pinyin_token.get(fp) ?? new Set();
			s.add(token_id);
			first_pinyin_token.set(fp, s);
		}
	}
}

await init_ctx();

console.log("初始化完毕");
