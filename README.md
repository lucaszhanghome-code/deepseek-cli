# deepseek-cli

A **dependency-free** terminal chat client for the [DeepSeek](https://platform.deepseek.com) API.
Pure Python standard library — no `pip install` of anything to run it, no vendor SDK, and
`urllib` is the only thing talking to the network.

```
   __                 __             __     ______ __
  / /____  ___  ___  / /_______  ___ / /__  / ___// /   /  _/
 / __/ _ \/ _ \/ _ \/ __/ ___/ _ \/ _ `/ _ \/ /__ / /    / /
 \__/\___/ .__/\___/\__/\__/  .__/\_,_/\___/\___//_/   /_/
        /_/                /_/
  deepseek-cli 0.1.0 - model deepseek-chat
  /help for commands, /exit to quit

you> explain token caching in one line
deepseek> Cache hits reuse previously computed prompt prefixes, so repeated
context is billed at a fraction of the miss rate.

[deepseek-chat - 18 in / 26 out - cache 0 hit / 18 miss - 1.2s]
```

## Features

- **Streaming by default** — replies appear token by token, with a spinner while you wait.
- **Bring your own key** — no key ships with the tool; on first run it asks you to paste
  yours, checks it against the API and saves it to your own config file.
- **Reasoning support** — `deepseek-reasoner` chain-of-thought is shown dimmed and separate
  from the answer (`--no-reasoning` hides it).
- **Persistent conversations** — every session is a JSON file you can grep, resume with
  `-c/--continue` or `--session ID`, and inspect with `--sessions`.
- **Interactive shell** — `/commands` for model switching, system prompts, temperature,
  undo, retry, token accounting and per-session cost estimates.
- **Scriptable** — pass a prompt, or pipe one in. `--json` emits structured output for
  jq/CI, and exit codes distinguish usage errors, API errors and interruptions.
- **Resilient** — automatic retries with exponential backoff for 429/5xx, honouring
  `Retry-After`, plus clear messages for auth and balance failures.
- **Works anywhere** — Windows, macOS and Linux; ANSI colour is detected and disabled
  automatically when output is piped.

## Install

Nothing is required to run it — Python 3.9+ is enough:

```console
$ git clone https://github.com/lucaszhanghome-code/deepseek-cli
$ cd deepseek-cli
$ python -m deepseek_cli "hello"
```

To get a `deepseek` command on your `PATH`:

```console
$ pip install .          # or: pipx install .
$ deepseek "hello"
```

On Windows you can also just run the bundled launcher, which forwards its arguments:

```console
> deepseek.cmd "hello"
```

On macOS and Linux the equivalent wrapper is `./deepseek`:

```console
$ ./deepseek "hello"
```

## Set your API key

Create a key at <https://platform.deepseek.com/api_keys>. The first time you run the
tool it notices that no key is configured and asks you to paste one:

```console
$ deepseek "hello"
No DeepSeek API key found.
Get one at https://platform.deepseek.com/api_keys, then paste it here (input is hidden).
It will be saved to ~/.deepseek-cli/config.json and sent only to the DeepSeek API.
API key:
API key accepted (sk-abc...wxyz) /home/you/.deepseek-cli/config.json
```

The pasted key is verified against `GET /user/balance` before it is stored, input is
hidden, and nothing is written unless the API accepts the key. From then on it is picked
up automatically — you are never asked twice.

The prompt is only shown on a real terminal with a prompt available. Non-interactive
callers (pipes, CI, `--json`, `--no-prompt`, `DEEPSEEK_CLI_NO_PROMPT=1`) get a clear error
and exit code `2` instead of hanging.

Prefer to configure it yourself? Any of these also work, and outrank the config file:

```console
$ export DEEPSEEK_API_KEY=sk-...        # PowerShell: $env:DEEPSEEK_API_KEY="sk-..."
$ python -m deepseek_cli --api-key sk-... "hello"
$ python -m deepseek_cli --set api_key=sk-...   # written to ~/.deepseek-cli/config.json
```

Inside the interactive shell, `/key sk-...` replaces the key, `/key` prompts for a new one
(also hidden and verified), and `/key` with no terminal prints the usage.

Nothing else needs configuring. Verify what the tool resolved:

```console
$ python -m deepseek_cli --print-config
$ # ... api key      sk-abc...wxyz (from config file)
```

> **Keep keys out of source control.** The config file lives in your home directory,
> not in the repo, and `.gitignore` excludes `config.json`, `.env` and `*.key`. When a key
> is saved the file is also created with owner-only permissions where the platform allows
> it. Only the masked form (`first6...last4`) is ever printed.

## Usage

### One-shot

```console
$ deepseek "summarise this error in one sentence: ..."
$ git diff | deepseek "write a commit message for this diff"
$ deepseek --json "list three colours" | jq -r .content
$ deepseek -o notes.md "add a short explanation of quadratic probing"
```

### Interactive

```console
$ deepseek
you> what is 2 + 2 in binary?
deepseek> 100
you> and the previous answer times 3?
deepseek> 1100
```

A trailing `\` continues a line, so you can paste multi-line prompts.

### Flags

| Flag | Purpose |
| --- | --- |
| `-m, --model` | `deepseek-chat` (default) or `deepseek-reasoner` |
| `-s, --system` | system prompt for the conversation |
| `-t, --temperature` | sampling temperature |
| `--max-tokens` | cap on generated tokens |
| `--no-stream` | wait for the complete reply |
| `--no-reasoning` | hide `reasoning_content` |
| `-c, --continue` | resume the most recent saved session |
| `--session ID` | resume a specific session |
| `--history-limit` | how many past messages to resend as context |
| `--no-save` | keep the conversation out of `~/.deepseek-cli/sessions` |
| `--api-key`, `--base-url` | override credentials or point at a proxy |
| `--no-prompt` | never ask for an API key (fail instead) |
| `--timeout`, `--max-retries` | request timeout and retry budget |
| `-o, --output FILE` | append the reply to a file |
| `--json` | structured output (one-shot mode) |
| `-q, --quiet`, `--no-color` | less chatter, no ANSI |
| `--list-models`, `--balance` | inspect the account |
| `--sessions`, `--print-config` | inspect local state |
| `--set KEY=VALUE` | persist a setting to `config.json` |

### Interactive commands

| Command | Purpose |
| --- | --- |
| `/help` | list every command |
| `/new`, `/clear` | start fresh, or empty the current conversation |
| `/retry`, `/undo`, `/history [n]` | re-send, remove, or review turns |
| `/model [id]`, `/models` | show, switch, or discover models |
| `/system [text\|-]` | show, set or clear the system prompt |
| `/temp [0-2]`, `/maxtokens [n]` | show or set generation parameters |
| `/stream on\|off`, `/reasoning on\|off` | toggle live streaming and reasoning output |
| `/usage`, `/balance`, `/config` | tokens, credit, effective settings |
| `/key [sk-...]` | show, replace and save the API key |
| `/save`, `/sessions`, `/load ID`, `/delete ID` | manage saved conversations |
| `/quit` (or `/exit`, Ctrl+D) | save and leave |

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | API or network failure |
| `2` | usage error (bad flag, missing prompt, missing or cancelled API key) |
| `130` | interrupted with Ctrl+C |

## Configuration

`~/.deepseek-cli/config.json` (override the location with `DEEPSEEK_CLI_HOME`):

```json
{
  "api_key": "",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com",
  "system_prompt": "",
  "temperature": 1.0,
  "max_tokens": 4096,
  "stream": true,
  "timeout": 120,
  "max_retries": 3,
  "history_limit": 40,
  "pricing": {}
}
```

Precedence is: command line flags → environment variables → `config.json` → defaults.

The API key follows the same order, and anything missing from all four triggers the
first-run prompt described above. Set `DEEPSEEK_CLI_NO_PROMPT=1` (or pass `--no-prompt`)
to turn that prompt off — useful in scripts and CI, where the tool will simply report the
missing key and exit `2`.

### Cost estimates

Prices change, so deepseek-cli never hard-codes them. Token counts are always reported;
add per-million-token rates to see `~$` figures too. `"*"` acts as a catch-all model key:

```console
$ deepseek --set 'pricing={"deepseek-chat": {"input_cache_hit": 0.028, "input_cache_miss": 0.28, "output": 0.42}}'
```

Check the current rates at <https://api-docs.deepseek.com/quick_start/pricing>.

## How it works

```
deepseek_cli/
  cli.py        argument handling, one-shot vs interactive dispatch, exit codes
  args.py       flag definitions and settings merge
  chat.py       the REPL: input handling, streaming, persistence
  commands.py   the /slash commands
  client.py     urllib transport, SSE parsing, retries, usage normalisation
  session.py    conversation state and JSON persistence
  config.py     config/credential resolution
  onboard.py    first-run API key prompt, verification and storage
  render.py     ANSI colours, spinner, tables, footers
```

The client exposes one generator, `DeepSeekClient.chat()`, that yields
`content`/`reasoning`/`usage`/`done` events. Streaming and non-streaming calls share that
path, so everything above it is transport-agnostic. Any OpenAI-compatible endpoint works
via `--base-url`.

## Development

```console
$ python -m unittest discover -s tests -t . -v
```

The suite runs fully offline: `tests/stub.py` starts a real HTTP server on a random port
that speaks the DeepSeek wire format (including SSE, 429 retries and error bodies), so no
API key or network access is needed. It passes on Linux, Windows and macOS with Python
3.9, 3.11 and 3.13.

## License

MIT — see [LICENSE](LICENSE).
