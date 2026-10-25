# Dataset licensing and attribution

This collection keeps each source license. It does not apply one new license to all documents.

- Wikipedia text: CC BY-SA 3.0 and GFDL, as stated by the pinned Wikimedia dataset card. Each record includes the article title and URL. The linked article and its revision history provide author attribution. The stored text is the original dataset text; token windows are excerpts. See https://creativecommons.org/licenses/by-sa/3.0/ and https://www.gnu.org/licenses/fdl-1.3.html.
- Mathlib4: Apache-2.0. Each record links to the exact source file and commit. The original file includes its author and copyright header.
- CPython: Python Software Foundation License Version 2, including the additional notices in the archived source LICENSE. Each record links to the exact file and commit. The original file text is preserved.

Full source license notices are in `../expanded_licenses/`. Dataset revision and archive hashes are in `manifest.json`. A shareable release must include this file, README.md, manifest.json, documents.jsonl, tokens.json, the configuration and these license notices. Tokens inherit the applicable source content terms. No model weights are part of this dataset release.
