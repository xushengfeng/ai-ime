import { assertEquals } from "@std/assert";
import { load_pinyin } from "../key_map/pinyin/gen_zi_pinyin.ts";
import { keys_to_pinyin } from "../key_map/pinyin/keys_to_pinyin.ts";
import { initLIME } from "../main.ts";

const { commit, single_ci } = await initLIME({ ziInd: load_pinyin() });

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
