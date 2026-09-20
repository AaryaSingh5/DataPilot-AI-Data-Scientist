from typing import Set
import networkx as nx
from datapilot.ledger.store import LedgerStore

class ProvenanceDAG:
    def __init__(self, store: LedgerStore, run_id: str):
        self.store = store
        self.run_id = run_id
        self._graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        
        cursor = self.store.conn.cursor()
        cursor.execute("SELECT id FROM evidence WHERE run_id = ?", (self.run_id,))
        rows = cursor.fetchall()
        
        for row in rows:
            ev_id = row['id']
            ev = self.store.get_evidence(ev_id)
            if ev:
                G.add_node(ev_id, kind=ev.kind, produced_by=ev.produced_by, columns=ev.columns)
                for dep in ev.depends_on:
                    G.add_edge(dep, ev_id)
                    
        return G

    def ancestors(self, ev_id: str) -> Set[str]:
        if ev_id not in self._graph:
            return set()
        return nx.ancestors(self._graph, ev_id)

    def descendants(self, ev_id: str) -> Set[str]:
        if ev_id not in self._graph:
            return set()
        return nx.descendants(self._graph, ev_id)

    def trace_to_columns(self, ev_id: str) -> Set[str]:
        columns = set()
        if ev_id in self._graph:
            columns.update(self._graph.nodes[ev_id].get('columns', []))
        for anc in self.ancestors(ev_id):
            columns.update(self._graph.nodes[anc].get('columns', []))
        return columns

    def to_networkx(self) -> nx.DiGraph:
        return self._graph
