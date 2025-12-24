import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "@std/cli/parse-args";
import { encodeBase64Url } from "@std/encoding/base64url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const KEY_FILE = path.join(__dirname, "key.txt");

/**
 * 生成随机 key（URL-safe）并返回字符串。
 */
function generateKey(nBytes: number = 32): string {
	const randomBytes = new Uint8Array(nBytes);
	crypto.getRandomValues(randomBytes);
	return encodeBase64Url(randomBytes);
}

/**
 * 对 key 做 SHA-256 并返回 hex 哈希。
 */
async function hashKey(key: string): Promise<string> {
	const encoder = new TextEncoder();
	const data = encoder.encode(key);
	const hashBuffer = await crypto.subtle.digest("SHA-256", data);

	// 转换为十六进制字符串
	const hashArray = Array.from(new Uint8Array(hashBuffer));
	return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * 将哈希追加到文件（每行一个），如果已存在则不重复写入。
 */
async function saveHash(
	hashHex: string,
	path: string = KEY_FILE,
): Promise<void> {
	// 读取已有哈希以避免重复
	const existing = new Set<string>();

	try {
		const content = await Deno.readTextFile(path);
		content.split("\n").forEach((line) => {
			const trimmed = line.trim();
			if (trimmed) existing.add(trimmed);
		});
	} catch (error) {
		if (!(error instanceof Deno.errors.NotFound)) {
			throw error;
		}
		// 文件不存在是正常情况，继续执行
	}

	if (existing.has(hashHex)) {
		return;
	}

	// 确保目录存在
	const dir = path.includes("/")
		? path.substring(0, path.lastIndexOf("/"))
		: ".";
	if (dir !== ".") {
		await Deno.mkdir(dir, { recursive: true });
	}

	// 追加哈希到文件
	await Deno.writeTextFile(path, `${hashHex}\n`, { append: true });
}

/**
 * 判断给定 key 的哈希是否存在于 key.txt 中。
 */
export async function verifyKey(
	key: string,
	path: string = KEY_FILE,
): Promise<boolean> {
	try {
		const hashHex = await hashKey(key);

		try {
			const content = await Deno.readTextFile(path);
			return content.split("\n").some((line) => line.trim() === hashHex);
		} catch (error) {
			if (error instanceof Deno.errors.NotFound) {
				return false;
			}
			throw error;
		}
	} catch (error) {
		console.error("验证 key 失败:", error);
		return false;
	}
}

async function main(): Promise<void> {
	const args = parseArgs(Deno.args, {
		string: ["verify"],
		alias: { verify: "v" },
	});

	if (args.verify) {
		const ok = await verifyKey(args.verify);
		if (ok) {
			console.log("Valid: key 的哈希存在于 key.txt。");
		} else {
			console.log("Invalid: key 的哈希不在 key.txt。");
		}
	} else {
		const key = generateKey();
		console.log("Generated key:", key);
		const h = await hashKey(key);
		await saveHash(h);
		console.log("Saved SHA-256 hash to", KEY_FILE);
	}
}

if (import.meta.main) {
	main().catch((error) => {
		console.error(error);
		Deno.exit(1);
	});
}
