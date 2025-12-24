import { type Context, Hono } from "hono";
import { bearerAuth } from "hono/bearer-auth";
import { HTTPException } from "hono/http-exception";
import { logger } from "hono/logger";
import { verifyKey } from "./key.ts";
import { load_pinyin } from "./key_map/pinyin/gen_zi_pinyin.ts";
import {
	keys_to_pinyin,
	type PinyinToKeyOptions,
} from "./key_map/pinyin/keys_to_pinyin.ts";
import { getUserData, initLIME } from "./main.ts";

const { single_ci, commit } = await initLIME({ ziInd: load_pinyin() });

const pinyinConfig: PinyinToKeyOptions = {
	shuangpin: true,
	fuzzy: {
		initial: {
			c: "ch",
			z: "zh",
			s: "sh",
			ch: "c",
			zh: "z",
			sh: "s",
			// 'l': 'n',
			// 'n': 'l',
			// 'f': 'h',
			// 'h': 'f',
			// 'r': 'l',
			// 'l': 'r',
		},
		final: {
			an: "ang",
			ang: "an",
			en: "eng",
			eng: "en",
			in: "ing",
			ing: "in",
			// "ian": "iang",
			// "iang": "ian",
			uan: "uang",
			uang: "uan",
		},
	},
};

const app = new Hono();

app.use("*", logger());
app.use(
	"/*",
	bearerAuth({
		verifyToken: (t) => {
			return verifyKey(t);
		},
	}),
);

// 辅助函数：处理查询参数
const getParam = (
	c: Context,
	key: string,
	defaultValue: string = "",
): string => {
	const value = c.req.query(key);
	return value ? decodeURIComponent(value) : defaultValue;
};

const getBoolParam = (
	c: Context,
	key: string,
	defaultValue: boolean = false,
): boolean => {
	const value = c.req.query(key);
	if (value === undefined) return defaultValue;
	return value === "true";
};

// API: 获取候选词 - POST 方法
app.post("/candidates", async (c) => {
	const body = await c.req.json<{ keys?: string }>();
	const keys = body.keys || "";

	console.log(keys);

	const pinyinInput = keys_to_pinyin(keys, pinyinConfig);
	const result = await single_ci(pinyinInput);

	return c.json(result);
});

// API: 获取候选词 - GET 方法
app.get("/candidates", async (c) => {
	const keys = getParam(c, "keys");

	console.log(keys);

	const pinyinInput = keys_to_pinyin(keys, pinyinConfig);
	const result = await single_ci(pinyinInput);

	return c.json(result);
});

// API: 提交文字 - POST 方法
app.post("/commit", async (c) => {
	try {
		const body = await c.req.json();
		const text = body.text || "";
		const isNew = body.new ?? true;
		const shouldUpdate = body.update ?? false;

		if (!text) {
			throw new HTTPException(400, { message: "未提供文本内容" });
		}

		commit(text, shouldUpdate, isNew);

		return c.json({
			message: "文本提交成功",
		});
	} catch (error) {
		if (error instanceof HTTPException) throw error;
		console.error("提交文本失败:", error);
		throw new HTTPException(400, { message: "请求数据格式错误" });
	}
});

// API: 提交文字 - GET 方法
app.get("/commit", (c) => {
	try {
		const text = getParam(c, "text");
		const isNew = getBoolParam(c, "new", true);
		const shouldUpdate = getBoolParam(c, "update", false);

		if (!text) {
			throw new HTTPException(400, { message: "未提供文本内容" });
		}

		commit(text, shouldUpdate, isNew);

		return c.json({
			message: "文本提交成功",
		});
	} catch (error) {
		if (error instanceof HTTPException) throw error;
		console.error("提交文本失败:", error);
		throw new HTTPException(400, { message: "请求参数错误" });
	}
});

app.get("/userdata", (c) => {
	return c.json(getUserData());
});

export default app;
