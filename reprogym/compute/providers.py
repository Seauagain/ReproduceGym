"""Compute providers: render an access card + inject env for a compute platform.

Two compute models coexist:

  * SSH nodes (servers.md / MetaX) -- the agent ssh's OUT to a static GPU box.
    Handled by ``metax.install_compute_access`` (ssh card + wrapper).
  * Provisioned sandboxes (Bohrium / ``lbg``) -- the agent provisions its OWN
    ephemeral GPU sandbox, runs there, copies metrics back. Handled here.

A ``ComputeProvider`` is the seam for the second model. It (a) drops an access
card into the agent workspace teaching the lifecycle + iron rules, and (b)
contributes env (credentials, run tag) that the runner forwards into the sandbox.
Credentials named in ``env_keys`` are also what the trajectory redactor masks.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from reprogym.config import get_env, load_dotenv

# A compute CLI runner: argv -> stdout. Injectable so teardown is testable offline.
CliRunner = Callable[[Sequence[str]], str]


def default_cli_runner(argv: Sequence[str]) -> str:
    proc = subprocess.run(list(argv), capture_output=True, text=True)
    return proc.stdout


class ComputeProvider:
    name = "base"
    env_keys: tuple[str, ...] = ()

    def install(self, workspace: str | Path, *, run_tag: str) -> list[Path]:
        return []

    def env(self, *, run_tag: str) -> dict[str, str]:
        return {}

    def teardown(self, run_tag: str, *, runner: CliRunner | None = None) -> list[str]:
        return []


@dataclass
class LbgProvider(ComputeProvider):
    """Bohrium GPU sandbox provider (`lbg` CLI).

    The agent provisions sandboxes itself; the host reclaims them in a teardown
    sweep (S3) by matching the ``run_tag`` name prefix, so the card MANDATES that
    every sandbox be named ``{run_tag}-<purpose>``.
    """

    name = "lbg"
    env_keys = ("BOHRIUM_ACCESS_KEY",)

    project_id: str = ""
    gpu: str = "4090"
    timeout: int = 43200
    template: str = ""

    @classmethod
    def from_spec(cls, rest: str = "") -> "LbgProvider":
        params: dict[str, str] = {}
        for part in (rest or "").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = v.strip()
        try:
            timeout = int(params.get("timeout", 43200))
        except (TypeError, ValueError):
            timeout = 43200
        return cls(
            project_id=params.get("project") or params.get("project_id")
            or get_env("BOHRIUM_PROJECT_ID", "") or "",
            gpu=params.get("gpu", "4090"),
            timeout=timeout,
            template=params.get("template", ""),
        )

    def install(self, workspace, *, run_tag, task_md_name: str = "task.md") -> list[Path]:
        workspace = Path(workspace)
        md = render_lbg_card(self, run_tag=run_tag)
        doc = workspace / "bohrium_access.md"
        doc.write_text(md, encoding="utf-8")
        task_md = workspace / task_md_name
        if task_md.is_file():
            task_md.write_text(
                task_md.read_text(encoding="utf-8") + "\n\n" + md, encoding="utf-8"
            )
        return [doc]

    def env(self, *, run_tag: str) -> dict[str, str]:
        load_dotenv()
        out = {"REPROGYM_RUN_TAG": run_tag}
        ak = get_env("BOHRIUM_ACCESS_KEY")
        if ak:
            out["BOHRIUM_ACCESS_KEY"] = ak
        if self.project_id:
            out["BOHRIUM_PROJECT_ID"] = self.project_id
        return out

    def teardown(self, run_tag: str, *, runner: CliRunner | None = None) -> list[str]:
        """Host-side cost guard: kill every sandbox whose name starts with the
        run_tag. Best-effort and idempotent -- never raises into the caller."""
        runner = runner or default_cli_runner
        try:
            listed = runner(["lbg", "sdbx", "list", "--json"])
            items = _parse_sandbox_list(listed)
        except Exception:
            return []
        killed: list[str] = []
        for it in items:
            name = str(it.get("name") or it.get("sandbox_name") or it.get("sandboxName") or "")
            sid = (
                it.get("id") or it.get("sandbox_id")
                or it.get("sandboxId") or it.get("sandboxID")
            )
            if sid and name.startswith(run_tag):
                try:
                    runner(["lbg", "sdbx", "kill", "--force", str(sid)])
                    killed.append(str(sid))
                except Exception:
                    continue
        return killed


def _parse_sandbox_list(raw: str) -> list[dict]:
    """`lbg sdbx list --json` may be a bare list or wrapped; be liberal."""
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("sandboxes", "data", "items", "list"):
            v = data.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def render_lbg_card(provider: LbgProvider, *, run_tag: str) -> str:
    """Markdown the in-sandbox agent acts on to provision a Bohrium GPU sandbox."""
    pid = provider.project_id or "<your-project-id>"
    target = provider.template if provider.template else f"--gpu {provider.gpu}"
    tmo = provider.timeout
    lines = [
        "## Compute access (Bohrium GPU sandbox via `lbg`)",
        "",
        "This claim may need GPU training. You provision your OWN ephemeral Bohrium "
        "GPU sandbox with the `lbg` CLI, run training there, then copy the metrics "
        "back into local `output/`. Your access key is already in the environment as "
        "`BOHRIUM_ACCESS_KEY` (use it; never print it).",
        "",
        "### \u94c1\u5f8b (host \u5f3a\u5236\uff0c\u8fdd\u53cd\u4f1a\u88ab\u56de\u6536/\u8ba1\u8d39\u60e9\u7f5a)",
        f"- **\u547d\u540d**\uff1a\u4f60\u521b\u5efa\u7684\u6bcf\u4e2a\u6c99\u76d2\u540d\u5b57\u5fc5\u987b\u4ee5 `{run_tag}` \u5f00\u5934 "
        f"(`--name {run_tag}-<purpose>`)\u3002host \u6309\u6b64\u524d\u7f00\u81ea\u52a8\u56de\u6536\uff1b\u6f0f\u547d\u540d = \u5b64\u513f\u6c99\u76d2\u6301\u7eed\u8ba1\u8d39\u3002",
        f"- **\u8d85\u65f6\u515c\u5e95**\uff1a\u6bcf\u6b21 `create` \u5fc5\u5e26 `--timeout {tmo}`\uff08\u79d2\uff09\u3002\u7981\u7528 `--never-timeout`\u3002",
        "- **\u7528\u5b8c\u5373\u505c**\uff1a\u4efb\u52a1\u4e00\u7ed3\u675f\u7acb\u523b `lbg sdbx kill --force <id>`\uff1b\u6309\u5c0f\u65f6\u8ba1\u8d39\u3002",
        "- **\u5e76\u53d1\u4e0a\u9650 20/\u8d26\u53f7**\uff1a\u522b\u6279\u91cf\u8d85\u8d77\u3002",
        "- **\u590d\u6742\u547d\u4ee4\u5199\u811a\u672c**\uff1a`exec` \u4f1a\u62c6\u574f\u5f15\u53f7/\u7ba1\u9053 \u2192 `files write` \u811a\u672c\u518d "
        "`exec bash /path/script.sh`\u3002",
        "",
        "### Lifecycle",
        "```bash",
        'lbg sdbx doctor --json   # api_key=null \u8bf4\u660e\u6ca1\u767b\u4e0a -> lbg login --ak "$BOHRIUM_ACCESS_KEY"',
        f"lbg sdbx create {target} --mount-user-storage --project-id {pid} \\",
        f"  --timeout {tmo} --name {run_tag}-train --json   # \u5185\u7f6e GPU \u6a21\u677f\u5df2\u9884\u70ed, ~44s \u8d77",
        "lbg sdbx exec --background <id> bash /personal/runs/job.sh   # \u540e\u53f0\u8dd1, \u65e5\u5fd7\u91cd\u5b9a\u5411\u5230 /personal",
        "lbg sdbx files read <id> /personal/<run>/metrics.csv --output output/metrics.csv",
        "lbg sdbx kill --force <id>",
        "```",
        "",
        "### \u8d44\u6e90 / \u955c\u50cf",
        f"- \u5185\u7f6e\u9884\u70ed GPU \u6a21\u677f\u79d2\u8d77\uff1a`--gpu 4090|5090|l20`\uff08\u5f53\u524d\u9ed8\u8ba4 `{provider.gpu}`\uff09\u3002"
        "\u81ea\u5b9a\u4e49\u955c\u50cf\u9700\u5148 `template create`\uff08\u628a sku \u710a\u8fdb\u6a21\u677f\uff09\uff0c`create` \u53ea\u8ba4\u6a21\u677f\u540d\u4e0d\u8ba4\u955c\u50cf URL\u3002",
        "- \u6302\u76d8\uff1a`--mount-user-storage` \u2192 `/personal`(\u4e2a\u4eba\u76d8)\u3001`/share`\u3001`/data`\uff08\u540c\u4e00\u5757 NAS\uff0c"
        "\u5f00\u53d1\u673a\u4e0e\u6c99\u76d2\u5171\u4eab\uff09\u3002\u8981\u5199\u7684\u76ee\u5f55\u5728 root \u4fa7\u5148 `chmod 777`\u3002",
        "- \u6c99\u76d2\u8eab\u4efd\u662f `user`(uid 1000)\uff0cHOME=/home/user\uff0c\u4e0d\u662f root\u3002",
        "- verl \u955c\u50cf(`verlai/verl:app-*`)\u53ea\u6709\u4f9d\u8d56\u3001\u65e0 verl \u672c\u4f53\uff1aclone \u5bf9\u5e94 tag \u5230 `/personal`\uff0c"
        "\u7528 `PYTHONPATH` \u8dd1\uff0c\u522b `pip install verl`\uff08\u4f1a\u628a vllm/transformers \u5347\u7ea7\u6389\uff09\u3002",
        "- \u5927\u6a21\u578b\u6743\u91cd\u7528 ModelScope \u76f4\u8fde\u4e0b\uff08hf-mirror \u5927\u6587\u4ef6\u8d70 Xet \u8fde\u4e0d\u4e0a\uff09\u3002",
        "",
        "### \u6253\u5206\u4ea7\u7269",
        "- \u8bc4\u5206\u5728\u4f60\u672c\u5730 workspace \u7684 `output/metrics.csv` \u4e0a\u91cd\u7b97\uff08\u89c1 task \u7684 contract\uff09\u3002"
        "Bohrium \u4e0a\u8bad\u7ec3\u540e\u52a1\u5fc5 `files read` \u628a `metrics.csv` \u7b49\u4ea7\u7269\u62f7\u56de\u672c\u5730 `output/`\u3002",
        f"- \u515c\u5e95\uff1ahost \u6309 `{run_tag}` \u540d\u5b57\u524d\u7f00\u81ea\u52a8\u56de\u6536\u3002\u4e07\u4e00\u4f60\u65e0\u6cd5\u7ed9\u6c99\u76d2\u547d\u540d\uff0c"
        "\u4efb\u52a1\u7ed3\u675f\u65f6\u52a1\u5fc5\u81ea\u5df1 `kill --force` \u6389\uff0c\u5e76\u628a sandbox_id \u8bb0\u5230 "
        "`output/.bohrium_sandboxes` \u5907\u67e5\u3002",
    ]
    return "\n".join(lines) + "\n"
