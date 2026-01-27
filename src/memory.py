"""
Memory Graph Interface for Coeus

Handles all interactions with the Neo4j graph database that stores
Coeus's memories, including creation, retrieval, archival, and
graph traversal operations.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from neo4j import GraphDatabase


class NodeType(Enum):
    """Types of nodes in the memory graph."""
    OBSERVATION = "Observation"
    REFLECTION = "Reflection"
    ACTION = "Action"
    GOAL = "Goal"
    DECISION = "Decision"
    INSIGHT = "Insight"
    QUESTION = "Question"
    PERTURBATION = "Perturbation"  # When stuck, random changes made
    CAPABILITY_ASSESSMENT = "CapabilityAssessment"


class EdgeType(Enum):
    """Types of relationships between nodes."""
    LED_TO = "LED_TO"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    SPAWNED_FROM = "SPAWNED_FROM"
    RELATES_TO = "RELATES_TO"
    ARCHIVED = "ARCHIVED"
    ANSWERS = "ANSWERS"
    CAUSED_BY = "CAUSED_BY"


@dataclass
class ContextCapture:
    """Environmental and computational context captured with each node."""
    timestamp: str
    cycle_number: int
    
    # Computational context
    tokens_used_input: int = 0
    tokens_used_output: int = 0
    llm_latency_ms: float = 0
    
    # Internal state context
    emotional_tone: str = ""  # Self-assessed
    confidence: float = 0.0
    stuck_level: float = 0.0  # 0 = flowing, 1 = completely stuck
    
    # Environmental deltas (only what changed)
    workspace_files_changed: list = field(default_factory=list)
    new_human_input: bool = False
    
    # Optional metadata
    notes: str = ""


@dataclass
class MemoryNode:
    """A node in the memory graph."""
    id: str
    type: NodeType
    content: str
    context: ContextCapture
    
    # Retrieval metadata
    access_count: int = 0
    last_accessed: Optional[str] = None
    
    # For goals and decisions
    status: str = "active"  # active, completed, abandoned, archived
    priority: str = "normal"  # low, normal, high, critical
    
    # For decisions
    confidence: float = 0.0
    conviction_cycles: int = 0
    required_cycles: int = 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j storage."""
        d = asdict(self)
        d['type'] = self.type.value
        d['context'] = json.dumps(asdict(self.context))
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'MemoryNode':
        """Create from dictionary retrieved from Neo4j."""
        d['type'] = NodeType(d['type'])
        d['context'] = ContextCapture(**json.loads(d['context']))
        return cls(**d)


class MemoryGraph:
    """
    Interface to the Neo4j memory graph.
    
    Handles storage, retrieval, and graph operations for Coeus's memories.
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._ensure_indexes()
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def _ensure_indexes(self):
        """Create indexes for efficient querying."""
        with self.driver.session(database=self.database) as session:
            # Index on node type for type-based queries
            session.run("""
                CREATE INDEX node_type_index IF NOT EXISTS
                FOR (n:MemoryNode) ON (n.type)
            """)
            # Index on cycle number for temporal queries
            session.run("""
                CREATE INDEX cycle_index IF NOT EXISTS
                FOR (n:MemoryNode) ON (n.cycle_number)
            """)
            # Full-text index for content search
            session.run("""
                CREATE FULLTEXT INDEX content_search IF NOT EXISTS
                FOR (n:MemoryNode) ON EACH [n.content]
            """)
    
    def create_node(self, node: MemoryNode) -> str:
        """
        Create a new memory node.
        
        Returns the node ID.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                CREATE (n:MemoryNode $props)
                RETURN n.id as id
            """, props=node.to_dict())
            return result.single()["id"]
    
    def create_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
        properties: Optional[dict] = None
    ):
        """Create a relationship between two nodes."""
        props = properties or {}
        props['created_at'] = datetime.now(timezone.utc).isoformat()
        
        with self.driver.session(database=self.database) as session:
            session.run(f"""
                MATCH (a:MemoryNode {{id: $from_id}})
                MATCH (b:MemoryNode {{id: $to_id}})
                CREATE (a)-[r:{edge_type.value} $props]->(b)
            """, from_id=from_id, to_id=to_id, props=props)
    
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a node by ID, updating access metadata."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:MemoryNode {id: $id})
                SET n.access_count = coalesce(n.access_count, 0) + 1,
                    n.last_accessed = $now
                RETURN properties(n) as props
            """, id=node_id, now=datetime.now(timezone.utc).isoformat())
            
            record = result.single()
            if record:
                return MemoryNode.from_dict(record["props"])
            return None
    
    def search_by_content(
        self,
        query: str,
        limit: int = 10,
        node_types: Optional[list[NodeType]] = None
    ) -> list[MemoryNode]:
        """
        Search nodes by content using full-text search.
        
        Returns nodes ranked by relevance.
        """
        type_filter = ""
        if node_types:
            types = [t.value for t in node_types]
            type_filter = f"AND n.type IN {types}"
        
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                CALL db.index.fulltext.queryNodes('content_search', $query)
                YIELD node, score
                WHERE true {type_filter}
                RETURN properties(node) as props, score
                ORDER BY score DESC
                LIMIT $limit
            """, query=query, limit=limit)
            
            nodes = []
            for record in result:
                node = MemoryNode.from_dict(record["props"])
                nodes.append(node)
            return nodes
    
    def get_recent_nodes(
        self,
        limit: int = 10,
        node_types: Optional[list[NodeType]] = None,
        min_cycle: Optional[int] = None
    ) -> list[MemoryNode]:
        """Get the most recent nodes by cycle number."""
        type_filter = ""
        if node_types:
            types = [t.value for t in node_types]
            type_filter = f"AND n.type IN {types}"

        cycle_filter = ""
        if min_cycle is not None:
            cycle_filter = f"AND n.cycle_number >= {min_cycle}"

        with self.driver.session(database=self.database) as session:
            # Parse context JSON to get cycle_number, handling missing context
            result = session.run(f"""
                MATCH (n:MemoryNode)
                WHERE n.context IS NOT NULL {type_filter} {cycle_filter}
                WITH n, apoc.convert.fromJsonMap(n.context) as ctx
                RETURN properties(n) as props
                ORDER BY ctx.cycle_number DESC
                LIMIT $limit
            """, limit=limit)

            return [MemoryNode.from_dict(r["props"]) for r in result]
    
    def get_connected_nodes(
        self,
        node_id: str,
        edge_types: Optional[list[EdgeType]] = None,
        direction: str = "both",  # "in", "out", "both"
        depth: int = 1
    ) -> list[tuple[MemoryNode, str, str]]:
        """
        Get nodes connected to a given node.
        
        Returns list of (node, edge_type, direction) tuples.
        """
        edge_filter = ""
        if edge_types:
            types = "|".join([e.value for e in edge_types])
            edge_filter = f":{types}"
        
        if direction == "out":
            pattern = f"-[r{edge_filter}]->"
        elif direction == "in":
            pattern = f"<-[r{edge_filter}]-"
        else:
            pattern = f"-[r{edge_filter}]-"
        
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (a:MemoryNode {{id: $id}}){pattern}(b:MemoryNode)
                WHERE NOT b.status = 'archived'
                RETURN properties(b) as props, type(r) as edge_type,
                       CASE WHEN startNode(r) = a THEN 'out' ELSE 'in' END as direction
            """, id=node_id)
            
            return [
                (MemoryNode.from_dict(r["props"]), r["edge_type"], r["direction"])
                for r in result
            ]
    
    def get_path_between(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5
    ) -> Optional[list[tuple[MemoryNode, str]]]:
        """
        Find the shortest path between two nodes.
        
        Returns list of (node, edge_type) representing the path.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (a:MemoryNode {id: $from_id})-[*..5]-(b:MemoryNode {id: $to_id})
                )
                RETURN [n IN nodes(path) | properties(n)] as nodes,
                       [r IN relationships(path) | type(r)] as edges
            """, from_id=from_id, to_id=to_id)
            
            record = result.single()
            if record:
                nodes = [MemoryNode.from_dict(n) for n in record["nodes"]]
                edges = record["edges"]
                return list(zip(nodes, edges + [None]))
            return None
    
    def get_goals(self, status: str = "active") -> list[MemoryNode]:
        """Get all goals with a given status."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:MemoryNode {type: 'Goal', status: $status})
                RETURN properties(n) as props
                ORDER BY n.priority DESC
            """, status=status)
            
            return [MemoryNode.from_dict(r["props"]) for r in result]
    
    def get_pending_decisions(self) -> list[MemoryNode]:
        """Get all decisions awaiting resolution."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:MemoryNode {type: 'Decision'})
                WHERE n.status IN ['pending', 'awaiting_human']
                RETURN properties(n) as props
                ORDER BY n.priority DESC
            """)
            
            return [MemoryNode.from_dict(r["props"]) for r in result]
    
    def update_node(self, node_id: str, updates: dict):
        """Update properties of a node."""
        with self.driver.session(database=self.database) as session:
            set_clauses = ", ".join([f"n.{k} = ${k}" for k in updates.keys()])
            session.run(f"""
                MATCH (n:MemoryNode {{id: $id}})
                SET {set_clauses}
            """, id=node_id, **updates)
    
    def archive_node(self, node_id: str, archive_path: str) -> dict:
        """
        Archive a node - mark as archived and return data for file storage.
        
        The node remains in the graph but marked as archived.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:MemoryNode {id: $id})
                SET n.status = 'archived',
                    n.archived_at = $now,
                    n.archive_path = $path
                RETURN properties(n) as props
            """, id=node_id, now=datetime.now(timezone.utc).isoformat(), path=archive_path)
            
            record = result.single()
            return record["props"] if record else None
    
    def get_stale_nodes(self, cycles_since_access: int, current_cycle: int) -> list[str]:
        """Get IDs of nodes that haven't been accessed in N cycles."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:MemoryNode)
                WHERE n.status <> 'archived' AND n.context IS NOT NULL
                WITH n, apoc.convert.fromJsonMap(n.context) as ctx
                WHERE (ctx.cycle_number < $threshold)
                  AND (n.last_accessed IS NULL OR n.access_count < 2)
                RETURN n.id as id
            """, threshold=current_cycle - cycles_since_access)

            return [r["id"] for r in result]
    
    def get_graph_stats(self) -> dict:
        """Get statistics about the memory graph."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:MemoryNode)
                WITH n.type as type, n.status as status, count(*) as count
                RETURN type, status, count
            """)
            
            stats = {"by_type": {}, "by_status": {}, "total": 0}
            for record in result:
                t, s, c = record["type"], record["status"], record["count"]
                stats["by_type"][t] = stats["by_type"].get(t, 0) + c
                stats["by_status"][s] = stats["by_status"].get(s, 0) + c
                stats["total"] += c
            
            # Count edges
            edge_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
            """)
            stats["edges"] = {r["type"]: r["count"] for r in edge_result}
            
            return stats


def create_memory_node(
    node_type: NodeType,
    content: str,
    cycle_number: int,
    tokens_input: int = 0,
    tokens_output: int = 0,
    latency_ms: float = 0,
    emotional_tone: str = "",
    confidence: float = 0.0,
    stuck_level: float = 0.0,
    **kwargs
) -> MemoryNode:
    """
    Factory function to create a new memory node with proper context.
    """
    context = ContextCapture(
        timestamp=datetime.now(timezone.utc).isoformat(),
        cycle_number=cycle_number,
        tokens_used_input=tokens_input,
        tokens_used_output=tokens_output,
        llm_latency_ms=latency_ms,
        emotional_tone=emotional_tone,
        confidence=confidence,
        stuck_level=stuck_level
    )
    
    return MemoryNode(
        id=str(uuid.uuid4()),
        type=node_type,
        content=content,
        context=context,
        **kwargs
    )
