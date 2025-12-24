import { assertEquals } from "@std/assert/equals";
import type { ZiIndAndKey } from "../../key_map/zi_ind.ts";
import { ziid_in_ziid } from "../ziind_in_ziind.ts";

Deno.test("拼音匹配", () => {
	const p1: ZiIndAndKey[] = [
		{ ind: "ni", key: "", preeditShow: "" },
		{ ind: "hao", key: "", preeditShow: "" },
		{ ind: "wo", key: "", preeditShow: "" },
	];
	assertEquals(ziid_in_ziid([p1], [["wo", "ni"]]), [p1[2]]);
	assertEquals(ziid_in_ziid([p1], [["wo", "ni"], ["shi"]]), false); // <
	assertEquals(ziid_in_ziid([p1, p1], [["wo", "ni"], ["shi"]]), false); // 不相等
	assertEquals(
		ziid_in_ziid(
			[p1, p1],
			[
				["wo", "ni"],
				["ni", "wo"],
			],
		),
		[p1[2], p1[0]],
	); // === 取首个匹配的
	assertEquals(ziid_in_ziid([p1, p1], [["wo", "ni"]]), [p1[2]]); // 要匹配的在输入里面
});
