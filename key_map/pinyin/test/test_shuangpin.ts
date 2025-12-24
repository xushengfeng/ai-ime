import { assertEquals } from "@std/assert";
import { generate_pinyin } from "../all_pinyin.ts";
import { generate_shuang_pinyin } from "../shuangpin.ts";

Deno.test("generate shuangpin", () => {
	const all = generate_pinyin();
	const shuangpin = generate_shuang_pinyin(all);
	console.log(all);
	console.log(shuangpin);
	const as = new Set(all);
	for (const i of Object.values(shuangpin)) {
		as.delete(i);
	}
	console.log(as);
	assertEquals(as.size, 0);
});
