import re
from typing import Any, Dict
from datapilot.ledger.store import LedgerStore

class UnresolvedPlaceholder(Exception):
    pass

class SafeResolver:
    # Whitelisted format specs
    ALLOWED_FORMATS = {
        ".0f", ".1f", ".2f", ".3f", ".4f",
        ".1%", ".2%", ",.0f", ",.2f", "s"
    }

    # Regex to match {ev_0001.result.pct_change:.1f} or {ev_0001.result.pct_change}
    PLACEHOLDER_REGEX = re.compile(r'\{([a-zA-Z0-9_]+)\.([a-zA-Z0-9_.]+)(?::([^\}]+))?\}')

    def __init__(self, store: LedgerStore):
        self.store = store

    def _traverse_dict(self, data: Dict[str, Any], path: str) -> Any:
        keys = path.split('.')
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise UnresolvedPlaceholder(f"Path '{path}' not found")
            current = current[key]
        return current

    def resolve_text(self, text: str) -> str:
        def repl(match: re.Match) -> str:
            ev_id = match.group(1)
            path = match.group(2)
            fmt = match.group(3)

            if fmt and fmt not in self.ALLOWED_FORMATS:
                raise UnresolvedPlaceholder(f"Format spec '{fmt}' is not allowed")

            ev = self.store.get_evidence(ev_id)
            if not ev:
                raise UnresolvedPlaceholder(f"Evidence '{ev_id}' not found")
            if ev.status != "ok":
                raise UnresolvedPlaceholder(f"Evidence '{ev_id}' has error status")

            # Allow paths like result.pct_change or params.alpha
            if path.startswith("result."):
                val = self._traverse_dict(ev.result, path[len("result."):])
            elif path.startswith("params."):
                val = self._traverse_dict(ev.params, path[len("params."):])
            else:
                raise UnresolvedPlaceholder(f"Path '{path}' must start with result. or params.")

            if fmt:
                if fmt == "s":
                    return str(val)
                # Ensure val is numeric before applying numeric formats
                if not isinstance(val, (int, float)):
                    raise UnresolvedPlaceholder(f"Value at '{path}' is not numeric, cannot apply format '{fmt}'")
                
                # Apply safe formatting
                return format(val, fmt)
            else:
                return str(val)

        return self.PLACEHOLDER_REGEX.sub(repl, text)
