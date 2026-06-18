#!/usr/bin/env node
// Version manager + release trigger.
//
// Keeps the version in sync across the three sources of truth, then commits,
// tags `v<version>`, and pushes. Pushing the tag triggers .github/workflows/build.yml,
// which builds the unsigned Windows installers and publishes the GitHub Release.
//
// Usage:
//   npm run release 0.2.0      # set an explicit version
//   npm run release patch      # 0.1.0 -> 0.1.1
//   npm run release minor      # 0.1.0 -> 0.2.0
//   npm run release major      # 0.1.0 -> 1.0.0
//   npm run release patch --dry-run   # show what would change, do nothing

import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const dryRun = args.includes("--dry-run");
const arg = args.find((a) => !a.startsWith("--"));

if (!arg) {
  console.error("Usage: npm run release <version|patch|minor|major> [--dry-run]");
  process.exit(1);
}

const pkgPath = join(root, "package.json");
const confPath = join(root, "src-tauri", "tauri.conf.json");
const cargoPath = join(root, "src-tauri", "Cargo.toml");

const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
const current = pkg.version;

function bump(version, kind) {
  const m = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!m) throw new Error(`Current version "${version}" is not x.y.z`);
  let [major, minor, patch] = m.slice(1).map(Number);
  if (kind === "major") return `${major + 1}.0.0`;
  if (kind === "minor") return `${major}.${minor + 1}.0`;
  if (kind === "patch") return `${major}.${minor}.${patch + 1}`;
  throw new Error(`Unknown bump kind: ${kind}`);
}

const next = ["patch", "minor", "major"].includes(arg) ? bump(current, arg) : arg;
if (!/^\d+\.\d+\.\d+$/.test(next)) {
  console.error(`Invalid version "${next}" — expected x.y.z`);
  process.exit(1);
}

console.log(`Version: ${current} -> ${next}${dryRun ? "  (dry run)" : ""}`);

// --- write the three version sources -------------------------------------
function setJsonVersion(path, version) {
  const obj = JSON.parse(readFileSync(path, "utf8"));
  obj.version = version;
  if (!dryRun) writeFileSync(path, JSON.stringify(obj, null, 2) + "\n");
}

function setCargoVersion(path, version) {
  const text = readFileSync(path, "utf8");
  // Replace only the first `version = "..."` (the [package] one at the top).
  const updated = text.replace(/^version = "[^"]*"/m, `version = "${version}"`);
  if (updated === text) throw new Error(`Could not find version in ${path}`);
  if (!dryRun) writeFileSync(path, updated);
}

setJsonVersion(pkgPath, next);
setJsonVersion(confPath, next);
setCargoVersion(cargoPath, next);
console.log("Updated: package.json, src-tauri/tauri.conf.json, src-tauri/Cargo.toml");

if (dryRun) {
  console.log("Dry run — no git changes made.");
  process.exit(0);
}

// --- commit, tag, push ----------------------------------------------------
function git(...a) {
  execFileSync("git", a, { cwd: root, stdio: "inherit" });
}

const tag = `v${next}`;
git("add", "package.json", "src-tauri/tauri.conf.json", "src-tauri/Cargo.toml");
git("commit", "-m", `chore: release ${tag}`);
git("tag", tag);
git("push");
git("push", "origin", tag);

console.log(`\nPushed ${tag}. GitHub Actions will build the installers and publish the release:`);
console.log("  https://github.com/johunsang/gemma-media-study/actions");
