import { join } from "@std/path/";

type OllamaOptions = {
	ollamaPath?: string;
};

function getDefaultOllamaPath(): string {
	const os = Deno.build.os;

	if (os === "darwin") {
		return join(Deno.env.get("HOME") || "", ".ollama", "models");
	} else if (os === "linux") {
		const userPath = join(Deno.env.get("HOME") || "", ".ollama", "models");
		try {
			if (Deno.statSync(userPath)?.isDirectory) {
				return userPath;
			}
		} catch {
			//
		}
		return "/usr/share/ollama/.ollama/models";
	} else if (os === "windows") {
		const userProfile = Deno.env.get("USERPROFILE") || "";
		return join(userProfile, ".ollama", "models");
	}

	throw new Error(`Unsupported operating system: ${os}`);
}

function getOllamaDir(basePath: string) {
	return {
		manifests: join(basePath, "manifests/registry.ollama.ai/library/"),
		models: join(basePath, "blobs/"),
	};
}

export function listOllamaModels(op?: OllamaOptions) {
	const basePath = op?.ollamaPath || getDefaultOllamaPath();
	const dirs = getOllamaDir(basePath);
	try {
		const manifests = Deno.readDirSync(dirs.manifests);
		const models: string[] = [];
		for (const modelName of manifests) {
			if (modelName.isDirectory) {
				for (const modelXinghao of Deno.readDirSync(
					join(dirs.manifests, modelName.name),
				)) {
					models.push(`${modelName.name}:${modelXinghao.name}`);
				}
			}
		}
		return models;
	} catch (error) {
		console.error("Error reading Ollama models:", error);
		return [];
	}
}

export function getOllamaModel(model: string, op?: OllamaOptions) {
	const basePath = op?.ollamaPath || getDefaultOllamaPath();
	const dirs = getOllamaDir(basePath);
	const manifestPath = join(dirs.manifests, model.replace(":", "/"));
	try {
		const manifestData = JSON.parse(Deno.readTextFileSync(manifestPath));
		const layer = manifestData.layers.find(
			(l: { mediaType: string }) =>
				l.mediaType === "application/vnd.ollama.image.model",
		);
		if (!layer) {
			console.error(`No model layer found for ${model}`);
			return undefined;
		}
		const modelBlobPath = join(
			dirs.models,
			layer.digest.replace("sha256:", "sha256-"),
		);
		try {
			if (Deno.statSync(modelBlobPath).isFile) {
				return modelBlobPath;
			} else {
				console.error(`Model blob path is not a file for ${model}`);
				return undefined;
			}
		} catch (error) {
			console.error(`Error accessing model blob path for ${model}:`, error);
			return undefined;
		}
	} catch (error) {
		console.error(`Error reading Ollama model manifest for ${model}:`, error);
		return undefined;
	}
}
