"""Stream CLI output while removing credential-bearing error URL parameters."""
import re
import subprocess
import sys
from pathlib import Path


def main():
    executable = Path(sys.executable).with_name("colab")
    process = subprocess.Popen([str(executable), *sys.argv[1:]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in process.stdout:
            line = re.sub(r"((?:colab-runtime-proxy-token|access_token|refresh_token|id_token|code_verifier)=)[^&\s\"'<>]+", r"\1[REDACTED]", line)
            line = re.sub(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", line)
            print(line, end="", flush=True)
        raise SystemExit(process.wait())
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise SystemExit(130)


if __name__ == "__main__":
    main()
