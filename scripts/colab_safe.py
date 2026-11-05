"""Stream CLI output while removing credential-bearing error URL parameters."""
import re
import subprocess
import sys
from pathlib import Path


def main():
    arguments = sys.argv[1:]
    if "exec" in arguments and "--config" in arguments:
        from colab_cli.state import StateStore
        config = arguments[arguments.index("--config") + 1]
        command = arguments[arguments.index("exec") + 1:]
        flag = "-s" if "-s" in command else "--session" if "--session" in command else None
        if flag:
            state = StateStore(config)
            session = state.get(command[command.index(flag) + 1])
            if session: state.add(session.model_copy(update={"session_id": None}))
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
