from datetime import datetime, timezone
import subprocess
import psutil
import os
import socket

def get_commit_id():
    git_result = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True)
    if git_result.returncode != 0:
        commit_id = "[No commit ID available]"
    else:
        commit_id = git_result.stdout.decode().strip()
    return commit_id

def get_metadata():
    now = datetime.now(timezone.utc).isoformat()
    metadata = {}
    metadata["comment"] = """ 
    This file was generated automatically.
    Do not modify it by hand."""
    metadata["time"] = now
    metadata["machine_name"] = socket.gethostname()
    metadata["analysis_code_version"] = get_commit_id()
    metadata["workflow_step"] = " ".join(psutil.Process(os.getpid()).cmdline())
    return metadata

def text_metadata(metadata, comment_char="#"):
    return "\n".join(
        [
            f"{comment_char} {k}: {v.replace('\n', f'\n{comment_char}')}"
            for k, v in (metadata).items()
        ]
    )

def provenance_string(comment_char="#"):
    provenance = text_metadata(get_metadata(), comment_char)
    return provenance+"\n"