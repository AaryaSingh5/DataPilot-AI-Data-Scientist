import sqlite3
import json
import hashlib
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from datapilot.ledger.models import Evidence

class LedgerStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    produced_by TEXT NOT NULL,
                    depends_on TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    code TEXT NOT NULL,
                    params TEXT NOT NULL,
                    result TEXT NOT NULL,
                    columns TEXT NOT NULL,
                    artifact_path TEXT,
                    artifact_hash TEXT,
                    lib_versions TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                )
            """)
            
            # Triggers to enforce append-only
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_evidence_update
                BEFORE UPDATE ON evidence
                BEGIN
                    SELECT RAISE(ABORT, 'Updates to evidence are not allowed.');
                END;
            """)
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_evidence_delete
                BEFORE DELETE ON evidence
                BEGIN
                    SELECT RAISE(ABORT, 'Deletions from evidence are not allowed.');
                END;
            """)

    def _get_last_hash(self, run_id: str) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT hash FROM evidence WHERE run_id = ? ORDER BY created_at DESC, id DESC LIMIT 1", (run_id,))
        row = cursor.fetchone()
        return row['hash'] if row else "0" * 64

    def _generate_next_id(self, run_id: str) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM evidence WHERE run_id = ?", (run_id,))
        count = cursor.fetchone()[0]
        return f"ev_{count + 1:04d}"

    def compute_hash(self, ev_dict: dict) -> str:
        # Exclude 'hash' itself and standardise JSON encoding
        hashable_dict = {k: v for k, v in ev_dict.items() if k != 'hash'}
        # Sort keys to ensure consistent hashing
        serialized = json.dumps(hashable_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def record_evidence(self, run_id: str, kind: str, produced_by: str, dataset_hash: str,
                        code: str, params: dict, result: dict, columns: list[str],
                        status: str, depends_on: list[str] = None, 
                        artifact_path: str = None, artifact_hash: str = None,
                        error: str = None, lib_versions: dict = None) -> Evidence:
        depends_on = depends_on or []
        lib_versions = lib_versions or {}
        
        prev_hash = self._get_last_hash(run_id)
        ev_id = self._generate_next_id(run_id)
        created_at = datetime.utcnow().isoformat()

        ev_dict = {
            "id": ev_id,
            "run_id": run_id,
            "kind": kind,
            "produced_by": produced_by,
            "depends_on": depends_on,
            "dataset_hash": dataset_hash,
            "code": code,
            "params": params,
            "result": result,
            "columns": columns,
            "artifact_path": artifact_path,
            "artifact_hash": artifact_hash,
            "lib_versions": lib_versions,
            "status": status,
            "error": error,
            "created_at": created_at,
            "prev_hash": prev_hash
        }
        
        ev_hash = self.compute_hash(ev_dict)
        ev_dict["hash"] = ev_hash

        with self.conn:
            self.conn.execute("""
                INSERT INTO evidence (
                    id, run_id, kind, produced_by, depends_on, dataset_hash, code,
                    params, result, columns, artifact_path, artifact_hash, lib_versions,
                    status, error, created_at, prev_hash, hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ev_dict["id"], ev_dict["run_id"], ev_dict["kind"], ev_dict["produced_by"],
                json.dumps(ev_dict["depends_on"]), ev_dict["dataset_hash"], ev_dict["code"],
                json.dumps(ev_dict["params"]), json.dumps(ev_dict["result"]), 
                json.dumps(ev_dict["columns"]), ev_dict["artifact_path"], ev_dict["artifact_hash"],
                json.dumps(ev_dict["lib_versions"]), ev_dict["status"], ev_dict["error"],
                ev_dict["created_at"], ev_dict["prev_hash"], ev_dict["hash"]
            ))

        return Evidence(**ev_dict)

    def get_evidence(self, ev_id: str) -> Optional[Evidence]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM evidence WHERE id = ?", (ev_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        ev_dict = dict(row)
        ev_dict["depends_on"] = json.loads(ev_dict["depends_on"])
        ev_dict["params"] = json.loads(ev_dict["params"])
        ev_dict["result"] = json.loads(ev_dict["result"])
        ev_dict["columns"] = json.loads(ev_dict["columns"])
        ev_dict["lib_versions"] = json.loads(ev_dict["lib_versions"])
        
        return Evidence(**ev_dict)

    def verify_chain(self, run_id: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM evidence WHERE run_id = ? ORDER BY id ASC", (run_id,))
        rows = cursor.fetchall()
        
        expected_prev_hash = "0" * 64
        for row in rows:
            ev_dict = dict(row)
            ev_dict["depends_on"] = json.loads(ev_dict["depends_on"])
            ev_dict["params"] = json.loads(ev_dict["params"])
            ev_dict["result"] = json.loads(ev_dict["result"])
            ev_dict["columns"] = json.loads(ev_dict["columns"])
            ev_dict["lib_versions"] = json.loads(ev_dict["lib_versions"])
            
            # Check prev_hash links correctly
            if ev_dict["prev_hash"] != expected_prev_hash:
                return False
            
            # Check current hash is valid
            computed = self.compute_hash(ev_dict)
            if computed != ev_dict["hash"]:
                return False
                
            expected_prev_hash = ev_dict["hash"]
            
        return True
