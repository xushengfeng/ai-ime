import path from "node:path";
import { fileURLToPath } from "node:url";
import { get_dict } from "../rime_dict.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function get_dict_py(filepath: string) {
	const d: Record<string, string[]> = {};
	const pa = path.join(__dirname, filepath);
	const ll = get_dict(pa);
	for (const i of ll) {
		const [zi, py, ..._r] = i.split("\t");
		if (!zi || !py) continue;
		const l = d[zi] ?? [];
		l.push(py);
		d[zi] = l;
	}
	return d;
}

export function load_pinyin() {
	const a = get_dict_py("../../assets/pinyin/8105.dict.yaml");
	const b = get_dict_py("../../assets/pinyin/41448.dict.yaml");
	const d: Record<string, Set<string>> = {};
	for (const i in a) {
		const l = d[i] ?? new Set();
		d[i] = l.union(new Set(a[i]));
	}
	for (const i in b) {
		const l = d[i] ?? new Set();
		d[i] = l.union(new Set(b[i]));
	}

	return {
		trans: (ci: string) => {
			const l: string[][] = [];
			for (const i of ci) {
				if (i in d) {
					l.push(Array.from(d[i]));
				} else {
					return [];
				}
			}
			return l;
		},
		allSymbol: new Set(Object.keys(a)).union(new Set(Object.keys(b))),
	};
}
