import os
import sys
import json
import uuid
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from datapilot.ledger.store import LedgerStore
from datapilot.sandbox.python_guard import PythonGuard, PythonGuardError

class PythonRunnerError(Exception):
    pass

class PythonRunner:
    def __init__(self, store: LedgerStore, snapshot_dir: Path, artifact_dir: Path, 
                 mode: str = "subprocess", require_network_isolation: bool = False,
                 timeout_sec: int = 15):
        self.store = store
        self.snapshot_dir = Path(snapshot_dir)
        self.artifact_dir = Path(artifact_dir)
        self.mode = mode
        self.require_network_isolation = require_network_isolation
        self.timeout_sec = timeout_sec
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.get("DATAPILOT_ENV", "development")
        if env == "production" and self.mode != "docker":
            raise PythonRunnerError("Docker sandbox mode is required in production environment.")

    def execute(self, run_id: str, dataset_hash: str, code: str, depends_on: List[str] = None) -> str:
        guard = PythonGuard()
        try:
            guard.check_code(code)
        except PythonGuardError as e:
            ev = self.store.record_evidence(
                run_id=run_id, kind="model", produced_by="ml", dataset_hash=dataset_hash,
                code=code, params={"sandbox_mode": self.mode}, result={}, columns=[], status="error", error=str(e),
                depends_on=depends_on
            )
            return ev.id

        if self.mode == "subprocess":
            return self._run_subprocess(run_id, dataset_hash, code, depends_on)
        else:
            return self._run_docker(run_id, dataset_hash, code, depends_on)

    def _run_subprocess(self, run_id: str, dataset_hash: str, code: str, depends_on: List[str]) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_file = Path(tmpdir) / "result.json"
            data_dir = self.snapshot_dir / dataset_hash
            
            # The child script sets up audit hooks, limits, and executes the user code
            child_code = f"""
import sys
import os
import json

def audit_hook(event, args):
    forbidden_events = {{"os.system", "subprocess.Popen", "socket.socket"}}
    if event in forbidden_events:
        raise RuntimeError(f"Audit hook blocked: {{event}}")

sys.addaudithook(audit_hook)

# Import blocker for builtin dangerous modules
import sys
class BlockedImporter:
    def find_spec(self, fullname, path, target=None):
        if fullname in {{"socket", "urllib", "requests", "subprocess", "os.system"}}:
            raise ImportError(f"Module {{fullname}} is blocked.")
        return None
sys.meta_path.insert(0, BlockedImporter())

# Attempt rlimits if available (Unix)
try:
    import resource
    # Example limits
    resource.setrlimit(resource.RLIMIT_CPU, ({self.timeout_sec}, {self.timeout_sec}))
    resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 1024*1024*1024)) # 1GB
    resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
except ImportError:
    pass

# User code namespace
namespace = {{
    "DATA_DIR": r"{str(data_dir.absolute())}",
    "RESULT_FILE": r"{str(result_file.absolute())}"
}}

try:
    exec({repr(code)}, namespace)
except Exception as e:
    with open(r"{str(result_file.absolute())}", "w") as f:
        json.dump({{"error": str(e)}}, f)
"""
            script_path = Path(tmpdir) / "script.py"
            script_path.write_text(child_code)

            cmd = [sys.executable, "-I", str(script_path)]
            
            if self.require_network_isolation and sys.platform != "win32":
                cmd = ["unshare", "--net"] + cmd
                
            # Scrub environment
            clean_env = {"PATH": os.environ.get("PATH", "")}
            
            try:
                # Process group to kill children if needed (Unix only)
                if sys.platform != "win32":
                    proc = subprocess.Popen(cmd, env=clean_env, cwd=tmpdir, preexec_fn=os.setsid, 
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        stdout, stderr = proc.communicate(timeout=self.timeout_sec)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(proc.pid), 9)
                        raise TimeoutError("Execution timed out.")
                else:
                    proc = subprocess.Popen(cmd, env=clean_env, cwd=tmpdir,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        stdout, stderr = proc.communicate(timeout=self.timeout_sec)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        raise TimeoutError("Execution timed out.")
                
                if proc.returncode != 0:
                    raise RuntimeError(f"Script failed: {{stderr}}")
                    
                if not result_file.exists():
                    raise RuntimeError("No result.json produced by script.")
                    
                # Size cap on result file (e.g. 5MB)
                if result_file.stat().st_size > 5 * 1024 * 1024:
                    raise RuntimeError("Result JSON exceeded size limit.")
                    
                with open(result_file, "r") as f:
                    result = json.load(f)
                    
                if "error" in result:
                    raise RuntimeError(result["error"])
                    
                ev = self.store.record_evidence(
                    run_id=run_id, kind="model", produced_by="ml", dataset_hash=dataset_hash,
                    code=code, params={"sandbox_mode": self.mode}, result=result, columns=[], status="ok",
                    depends_on=depends_on
                )
                return ev.id
                
            except Exception as e:
                ev = self.store.record_evidence(
                    run_id=run_id, kind="model", produced_by="ml", dataset_hash=dataset_hash,
                    code=code, params={"sandbox_mode": self.mode}, result={}, columns=[], status="error", error=str(e),
                    depends_on=depends_on
                )
                return ev.id

    def _run_docker(self, run_id: str, dataset_hash: str, code: str, depends_on: List[str]) -> str:
        # Stub for docker execution
        ev = self.store.record_evidence(
            run_id=run_id, kind="model", produced_by="ml", dataset_hash=dataset_hash,
            code=code, params={"sandbox_mode": "docker"}, result={}, columns=[], status="error", 
            error="Docker mode not yet implemented",
            depends_on=depends_on
        )
        return ev.id
