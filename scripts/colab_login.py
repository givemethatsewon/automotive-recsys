"""Open Google's native browser consent flow without printing auth URLs/codes.

Credentials are isolated from other gcloud/MCP workflows and from the Git repo.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    os.umask(0o077)
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    config = root / "automotive-recsys" / "colab" / "gcloud"
    config.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.chmod(0o700)
    command = shutil.which("gcloud")
    if command is None:
        raise SystemExit("Google Cloud CLI (gcloud) is required for browser sign-in.")
    scopes = ",".join([
        "openid", "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/colaboratory",
    ])
    environment = os.environ | {"CLOUDSDK_CONFIG": str(config)}
    print("Opening Google sign-in in your browser. Select the account holding your Colab compute units.", flush=True)
    process = subprocess.Popen(
        [command, "auth", "application-default", "login", f"--scopes={scopes}", "--disable-quota-project"],
        env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    try:
        for line in process.stdout:
            # Keep OAuth state and authorization responses out of tool/chat logs.
            print(re.sub(r"https?://\S+", "[Google sign-in link hidden]", line), end="", flush=True)
        status = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise SystemExit(130)
    finally:
        for file in config.rglob("*"):
            if file.is_file():
                file.chmod(0o600)
    if status:
        raise SystemExit(status)
    print("Colab CLI credentials saved with owner-only permissions outside the project.", flush=True)


if __name__ == "__main__":
    main()
