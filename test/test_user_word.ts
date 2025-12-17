import { assertEquals } from "jsr:@std/assert@1.0.16";
import { keys_to_pinyin } from "../key_map/pinyin/keys_to_pinyin.ts";
import { commit, single_ci } from "../main.ts";

Deno.test("组词", async () => {
	commit("冰灯");
	commit("灯盏");
	commit("悠然自得");
	commit("冰灯");
	const r = await single_ci(keys_to_pinyin("bingdeng"));
	console.log(r.candidates.slice(0, 5));
	assertEquals(r.candidates[0].word, "冰灯");
});

Deno.test("智能组词", async () => {
	commit("冰灯");
	commit("是");
	commit("流行于");
	const nr = await single_ci(keys_to_pinyin("vsgobz", { shuangpin: true }));
	console.log(nr.candidates.slice(0, 5));
});
