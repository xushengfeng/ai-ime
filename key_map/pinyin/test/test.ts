import { assertEquals } from "@std/assert";
import { assert } from "@std/assert/assert";
import { generate_fuzzy_pinyin } from "../fuzzy_pinyin.ts";
import { load_pinyin } from "../gen_zi_pinyin.ts";
import { keys_to_pinyin } from "../keys_to_pinyin.ts";
import { spilt_pinyin } from "../split_pinyin.ts";

Deno.test("split pinyin", () => {
	assertEquals(spilt_pinyin("ni"), ["n", "i"]);
	assertEquals(spilt_pinyin("zhe"), ["zh", "e"]);
	assertEquals(spilt_pinyin("zhang"), ["zh", "ang"]);
});

Deno.test("fuzzy pinyin", () => {
	assertEquals(
		new Set(generate_fuzzy_pinyin("shang")),
		new Set(["shang", "shan", "san", "sang"]),
	);
});

Deno.test("拼音 常规", () => {
	const x = keys_to_pinyin("nihao");
	assertEquals(x, [
		[{ ind: "ni", key: "ni", preeditShow: "ni" }],
		[{ ind: "hao", key: "hao", preeditShow: "hao" }],
	]);
});

Deno.test("拼音 部分", () => {
	const x = keys_to_pinyin("nihaow").at(-1);
	assert((x?.length ?? 0) > 0);
	const x1 = x?.find((v) => v.ind === "wo");
	assertEquals(x1, { key: "w", ind: "wo", preeditShow: "w" });
});

Deno.test("拼音 部分2", () => {
	const x = keys_to_pinyin("nihao", { shuangpin: true }).at(-1);
	assert((x?.length ?? 0) > 0);
	const x1 = x?.find((v) => v.ind === "ou");
	assertEquals(x1, { key: "o", ind: "ou", preeditShow: "o" });
});

Deno.test("拼音 部分3", () => {
	const x = keys_to_pinyin("a").at(-1);
	assert((x?.length ?? 0) > 0);
	assertEquals(
		x?.find((v) => v.ind === "a"),
		{ key: "a", ind: "a", preeditShow: "a" },
	);
	assertEquals(
		x?.find((v) => v.ind === "ai"),
		{ key: "a", ind: "ai", preeditShow: "a" },
	);
	const x1 = keys_to_pinyin("tma", { shuangpin: true }).at(-1);
	assert((x1?.length ?? 0) > 0);
	assertEquals(
		x1?.find((v) => v.ind === "a"),
		{ key: "a", ind: "a", preeditShow: "a" },
	);
	assertEquals(
		x1?.find((v) => v.ind === "ai"),
		{ key: "a", ind: "ai", preeditShow: "a" },
	);
	const x2 = keys_to_pinyin("tmaa", { shuangpin: true }).at(-1);
	assert((x2?.length ?? 0) > 0);
	console.log(x2);

	assertEquals(x2?.at(-1), { key: "aa", ind: "a", preeditShow: "a" });
});

Deno.test("拼音 分隔符", () => {
	const x = keys_to_pinyin("ni'");
	assertEquals(x, [[{ key: "ni'", ind: "ni", preeditShow: "ni" }]]);
	const x1 = keys_to_pinyin("ni'hao'wo");
	assertEquals(x1, [
		[{ key: "ni'", ind: "ni", preeditShow: "ni" }],
		[{ key: "hao'", ind: "hao", preeditShow: "hao" }],
		[{ key: "wo", ind: "wo", preeditShow: "wo" }],
	]);
});

Deno.test("双拼", () => {
	const x = keys_to_pinyin("xxxx", { shuangpin: true });
	assertEquals(x, [
		[{ key: "xx", ind: "xie", preeditShow: "xie" }],
		[{ key: "xx", ind: "xie", preeditShow: "xie" }],
	]);
});

Deno.test("字转化拼音", () => {
	const pinyin = load_pinyin();
	const x = pinyin.trans("你好");
	assertEquals(x, [["ni"], ["hao"]]);
});
