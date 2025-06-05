"""
SynapseMemory: Human-like long-term and short-term memory for LLM chatbots.
Includes skip, sleep (with user notification), approximate k-NN graph, summary-vector cache,
and a unified handle_message entry point.
"""

import os
import sqlite3
import threading
import datetime
import numpy as np
import faiss
import networkx as nx
import yaml

from sentence_transformers import SentenceTransformer
from datetime import datetime as dt
import re  # for robust number extraction from LLM response


class SynapseMemory:
    """Integrates long-term and short-term memory with human-like recall."""

    def __init__(self, config_path: str = "sm_config.yaml", llm_callback=None):
        # ==== CONFIGURATION ====
        if not os.path.exists(config_path):
            default_conf = {
                "embedding_model": "all-MiniLM-L6-v2",
                "faiss_top_k_embed": 30,
                "graph_sim_threshold": 0.7,
                "graph_nn_k": 10,
                "scoring_weights": {
                    "sim_score": 0.35,
                    "context_score": 0.15,
                    "recency_score": 0.15,
                    "imp_score": 0.10,
                    "freq_score": 0.10,
                    "centrality_score": 0.15
                },
                "context_window_pairs": 6,
                "db": {"type": "sqlite", "sqlite_path": "memory.db"},
                "sleep": {"n_clusters": 6, "semantic_threshold": 0.85, "time_window_sec": 600}
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(default_conf, f)

        with open(config_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)

        self.embedding_model_name = conf.get("embedding_model", "")
        self.faiss_top_k_embed = int(conf.get("faiss_top_k_embed", 30))
        self.graph_sim_threshold = float(conf.get("graph_sim_threshold", 0.7))
        self.graph_nn_k = int(conf.get("graph_nn_k", 10))
        self.scoring_weights = conf.get("scoring_weights", {})
        self.context_window_pairs = int(conf.get("context_window_pairs", 6))

        db_conf = conf.get("db", {})
        self.db_path = db_conf.get("sqlite_path", "memory.db")

        sleep_conf = conf.get("sleep", {})
        self.sleep_n_clusters = int(sleep_conf.get("n_clusters", 6))
        self.sleep_semantic_threshold = float(sleep_conf.get("semantic_threshold", 0.85))
        self.sleep_time_window = int(sleep_conf.get("time_window_sec", 600))

        # ==== LLM CALLBACK ====
        self._llm_callback = llm_callback

        # ==== EMBEDDING & INDEX ====
        try:
            self.embedder = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
        self.index = None
        self._thread_local = threading.local()

        # ==== GRAPH & CACHE ====
        self.graph = nx.Graph()
        self.centrality_scores = {}
        self.summary_embeddings = {}  # id -> embedding vector
        self.summary_norms = {}       # id -> normalized embedding

        # ==== SETUP DATABASE & INDEX ====
        conn_main = sqlite3.connect(self.db_path, timeout=30)
        self._ensure_tables_exist(conn_main)
        conn_main.close()

        self._load_faiss_index_if_exists()
        self._build_graph_and_centrality()

    def _get_connection(self):
        """Return a thread-local SQLite connection."""
        if not hasattr(self._thread_local, "conn"):
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._thread_local.conn = conn
        return self._thread_local.conn

    def _ensure_tables_exist(self, conn):
        """Create required tables if they do not exist."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                title            TEXT,
                title_embedding  BLOB,
                content          TEXT,
                summary_text     TEXT,
                timestamp        TEXT,
                importance       REAL DEFAULT 1.0,
                processed        INTEGER DEFAULT 1,
                recall_count     INTEGER DEFAULT 0
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_cluster (
                cluster_id      INTEGER,
                experience_id   INTEGER,
                cluster_label   TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experience_relations (
                exp_id_a      INTEGER,
                exp_id_b      INTEGER,
                relation_type TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sleep_history (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_start      TEXT,
                cycle_end        TEXT,
                original_count   INTEGER,
                summary_text     TEXT,
                notes            TEXT
            );
        """)
        conn.commit()

    @property
    def index_file(self):
        """Path to FAISS index file."""
        return os.path.splitext(self.db_path)[0] + "_index.faiss"

    def _load_faiss_index_if_exists(self):
        """Load existing FAISS index if present."""
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
            except Exception:
                self.index = None

    def add_experience(self, title: str, content: str, summary_text: str,
                       importance: float = 1.0, processed: int = 1) -> int:
        """
        Insert a new experience: generate embedding, store in DB, update FAISS and graph.
        Returns the new experience ID.
        """
        conn = self._get_connection()
        cur = conn.cursor()

        # Embed content (use same vector for title_embedding)
        vec = self.embedder.encode(content).astype("float32")
        vec_blob = vec.tobytes()
        ts = dt.utcnow().isoformat()

        try:
            cur.execute("""
                INSERT INTO experiences
                (title, title_embedding, content, summary_text,
                 timestamp, importance, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                title,
                sqlite3.Binary(vec_blob),
                content,
                summary_text,
                ts,
                importance,
                processed
            ))
            exp_id = cur.lastrowid
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"DB insert failed: {e}")

        # Update FAISS index
        try:
            if self.index is None:
                dim = vec.shape[0]
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
            self.index.add_with_ids(np.array([vec]), np.array([exp_id], dtype="int64"))
            faiss.write_index(self.index, self.index_file)
        except Exception:
            pass

        # Rebuild graph/caches
        self._build_graph_and_centrality()
        return exp_id

    def _build_graph_and_centrality(self):
        """
        Build an approximate k-NN graph on summary embeddings,
        compute degree centrality, and cache summary embeddings.
        """
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, summary_text FROM experiences")
        rows = cur.fetchall()
        n = len(rows)
        if n == 0:
            self.graph = nx.Graph()
            self.centrality_scores = {}
            self.summary_embeddings.clear()
            self.summary_norms.clear()
            return

        ids = [r["id"] for r in rows]
        texts = [r["summary_text"] for r in rows]
        embs = self.embedder.encode(texts, convert_to_numpy=True).astype("float32")
        norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

        # Cache embeddings and norms
        self.summary_embeddings = {eid: emb for eid, emb in zip(ids, embs)}
        self.summary_norms = {eid: norm for eid, norm in zip(ids, norms)}

        dim = norms.shape[1]
        idx_ip = faiss.IndexFlatIP(dim)
        idx_ip.add(norms)
        id_map = {i: eid for i, eid in enumerate(ids)}

        G = nx.Graph()
        G.add_nodes_from(ids)
        k = min(self.graph_nn_k + 1, n)
        D, I = idx_ip.search(norms, k)

        for i in range(n):
            src = ids[i]
            for neigh_idx, sim in zip(I[i, 1:], D[i, 1:]):
                if sim >= self.graph_sim_threshold:
                    dst = id_map[neigh_idx]
                    G.add_edge(src, dst)

        self.graph = G
        self.centrality_scores = nx.degree_centrality(G)

    def extract_context_from_tracker(self, tracker, max_pairs: int) -> list:
        """Extract last max_pairs of user/bot messages from a Rasa Tracker."""
        events = tracker.events
        msgs = []
        for e in reversed(events):
            if e.get("event") == "user":
                msgs.append(f"User: {e.get('text')}")
            elif e.get("event") == "bot":
                msgs.append(f"Bot: {e.get('text')}")
            if len(msgs) >= max_pairs * 2:
                break
        return list(reversed(msgs))

    def recall(self, query: str, context_messages: list, top_k: int = 3) -> str:
        """
        1. Embed query, FAISS search for candidates.
        2. Compute context embedding.
        3. Score candidates by sim, context, recency, importance, freq, centrality.
        4. Prompt LLM with system instructions + options (0=No recall, 1=Sleep, 2+=titles).
        5. Parse LLM response robustly, dispatch to NoRecall, Sleep, or normal flow.
        6. If Sleep, run sleep_cycle and then generate user-facing notice via LLM.
        """
        # 1. Query embedding & FAISS search
        q_vec = self.embedder.encode(query).astype("float32")
        if self.index is None or (hasattr(self.index, "ntotal") and self.index.ntotal == 0):
            raise RuntimeError("No experiences available.")
        distances, ids = self.index.search(np.array([q_vec]), self.faiss_top_k_embed)
        dists, ids = distances[0], ids[0]

        conn = self._get_connection()
        cur = conn.cursor()

        # 2. Context embedding
        max_msgs = self.context_window_pairs * 2
        rec_msgs = context_messages[-max_msgs:] if len(context_messages) > max_msgs else context_messages
        ctx_text = "\n".join(rec_msgs)
        try:
            ctx_vec = self.embedder.encode(ctx_text).astype("float32")
            ctx_norm = ctx_vec / (np.linalg.norm(ctx_vec) + 1e-12)
        except Exception:
            ctx_norm = np.zeros((1,), dtype="float32")

        # 3. Max recall_count for normalization
        cur.execute("SELECT MAX(recall_count) as max_cnt FROM experiences")
        row_max = cur.fetchone()
        max_recall = row_max["max_cnt"] or 1

        candidates = []
        now = dt.utcnow()
        for dist, eid in zip(dists, ids):
            if eid < 0:
                continue
            cur.execute("""
                SELECT title, summary_text, timestamp, importance, recall_count
                FROM experiences WHERE id = ?
            """, (int(eid),))
            row = cur.fetchone()
            if not row:
                continue
            title, summ, ts_str, imp, freq = row

            sim_score = 1.0 / (1.0 + float(dist))

            summary_norm = self.summary_norms.get(eid)
            if summary_norm is not None and ctx_norm.shape == summary_norm.shape:
                cos_ctx = float(np.dot(ctx_norm, summary_norm))
                context_score = max(0.0, min(1.0, cos_ctx))
            else:
                context_score = 0.0

            try:
                ts = dt.fromisoformat(ts_str)
            except Exception:
                ts = now - datetime.timedelta(days=3650)
            delta_days = (now - ts).total_seconds() / (3600 * 24)
            recency_score = np.exp(-delta_days / 1.0)

            imp_score = min(max((imp - 1.0) / 9.0, 0.0), 1.0)
            freq_score = np.log1p(freq) / np.log1p(max_recall)
            centrality_score = float(self.centrality_scores.get(eid, 0.0))

            w = self.scoring_weights
            total = (
                w["sim_score"] * sim_score +
                w["context_score"] * context_score +
                w["recency_score"] * recency_score +
                w["imp_score"] * imp_score +
                w["freq_score"] * freq_score +
                w["centrality_score"] * centrality_score
            )

            candidates.append({
                "id": int(eid),
                "title": title,
                "summary": summ,
                "total_score": total
            })

        if not candidates:
            raise RuntimeError("No candidates found.")

        candidates.sort(key=lambda x: x["total_score"], reverse=True)
        top_cands = candidates[:top_k]

        # 4. Build prompt with updated system instruction
        system_prompt = (
            "You are an assistant that can retrieve relevant memories from the system as needed, "
            "and upon user request, select \"Sleep memory for now\" to optimize your memory.\n"
            "You will be shown a numbered list of options.\n"
            "Respond with only the single digit corresponding to your choice, without any extra text.\n"
            "Do not output anything else."
        )
        option_lines = [
            "0. No recall needed",
            "1. Sleep memory for now"
        ]
        for i, c in enumerate(top_cands, start=2):
            option_lines.append(f"{i}. {c['title']}")
        option_lines.append(f"User question: {query}")
        full_prompt = system_prompt + "\n" + "\n".join(option_lines)

        if not self._llm_callback:
            raise RuntimeError("LLM callback not set.")
        raw_resp = self._llm_callback(full_prompt)

        # 5. Robustly extract the first integer from LLM response
        m = re.search(r"\d+", raw_resp or "")
        choice = int(m.group()) if m else 0

        # 6. Dispatch based on choice
        if choice == 1:
            # Sleep chosen: run sleep_cycle and then generate notification
            sleep_result = self.sleep_cycle(llm_summary_fn=self._llm_callback)
            notify_system_prompt = (
                "You are the assistant. Inform the user that memory is going to sleep now, "
                "and that you will improve things in the background."
            )
            notify_full = notify_system_prompt + "\nUser request: Please notify me."
            user_notice = self._llm_callback(notify_full)
            return user_notice.strip() + "\n" + sleep_result.get("message", "").strip()

        if choice == 0:
            # No recall: ask LLM using only context
            no_recall_prompt = f"Recent conversation:\n{ctx_text}\nUser question: {query}\nAnswer:"
            return self._llm_callback(no_recall_prompt)

        # 7. Normal recall flow
        chosen = top_cands[choice - 2]
        cid = chosen["id"]
        csum = chosen["summary"]

        try:
            cur.execute("UPDATE experiences SET recall_count = recall_count + 1 WHERE id = ?;", (cid,))
            conn.commit()
        except Exception:
            conn.rollback()

        cur.execute("SELECT id FROM experiences WHERE processed = 1 ORDER BY id")
        all_ids = [r["id"] for r in cur.fetchall()]
        pos = all_ids.index(cid) if cid in all_ids else -1

        prev_ids = all_ids[:pos][::-1] if pos >= 0 else []
        next_ids = all_ids[pos + 1:] if pos >= 0 else []

        MAX_EXTRA = 100
        rem = MAX_EXTRA
        extra = ""
        step = 0
        pi, ni = 0, 0
        while rem > 0 and (pi < len(prev_ids) or ni < len(next_ids)):
            if step % 2 == 0 and pi < len(prev_ids):
                pid = prev_ids[pi]
                cur.execute("SELECT summary_text FROM experiences WHERE id = ?", (pid,))
                text = cur.fetchone()["summary_text"]
                if len(text) <= rem:
                    extra = text + extra
                    rem -= len(text)
                    pi += 1
                else:
                    break
            elif step % 2 == 1 and ni < len(next_ids):
                nid = next_ids[ni]
                cur.execute("SELECT summary_text FROM experiences WHERE id = ?", (nid,))
                text = cur.fetchone()["summary_text"]
                if len(text) <= rem:
                    extra = extra + text
                    rem -= len(text)
                    ni += 1
                else:
                    break
            else:
                if pi < len(prev_ids):
                    pid = prev_ids[pi]
                    cur.execute("SELECT summary_text FROM experiences WHERE id = ?", (pid,))
                    text = cur.fetchone()["summary_text"]
                    if len(text) <= rem:
                        extra = text + extra
                        rem -= len(text)
                        pi += 1
                    else:
                        break
                elif ni < len(next_ids):
                    nid = next_ids[ni]
                    cur.execute("SELECT summary_text FROM experiences WHERE id = ?", (nid,))
                    text = cur.fetchone()["summary_text"]
                    if len(text) <= rem:
                        extra = extra + text
                        rem -= len(text)
                        ni += 1
                    else:
                        break
                else:
                    break
            step += 1

        final_prompt = (
            f"Using the following summary information, please answer:\n\n"
            f"{csum}\n\n{extra}\n\nUser question: {query}\nAnswer:"
        )
        return self._llm_callback(final_prompt)

    def sleep_cycle(self, llm_summary_fn=None, rebuild_index: bool = True) -> dict:
        """
        Perform a sleep cycle:
        - Cluster recent experiences (within time_window)
        - Generate summaries for clusters
        - Save to sleep_history
        - Optionally rebuild FAISS index
        """
        if llm_summary_fn is None:
            raise RuntimeError("LLM summary function not set.")

        conn = self._get_connection()
        cur = conn.cursor()
        window_start_ts = dt.utcnow().timestamp() - self.sleep_time_window
        window_start_iso = dt.fromtimestamp(window_start_ts).isoformat()
        cur.execute(
            "SELECT id, content FROM experiences WHERE timestamp >= ?",
            (window_start_iso,)
        )
        rows = cur.fetchall()
        if not rows:
            return {"message": "Sleep Cycle: no recent data.", "latest_sleep_history": None}

        ids = []
        mats = []
        for rec_id, content in rows:
            vec = self.embedder.encode(content).astype("float32")
            ids.append(int(rec_id))
            mats.append(vec)
        data_matrix = np.stack(mats)
        d = data_matrix.shape[1]

        kmeans = faiss.Kmeans(d, self.sleep_n_clusters, niter=20, verbose=False)
        kmeans.train(data_matrix)
        cluster_ids = kmeans.index.search(data_matrix, 1)[1].reshape(-1)

        cluster_map = {}
        for rec_id, cid in zip(ids, cluster_ids):
            cluster_map.setdefault(int(cid), []).append(int(rec_id))

        summary_text_for_history = ""
        for cid, exp_list in cluster_map.items():
            combined_text = ""
            for eid in exp_list:
                cur.execute("SELECT content FROM experiences WHERE id = ?", (eid,))
                row2 = cur.fetchone()
                if row2:
                    combined_text += row2[0] + "\n"
            summary_prompt = f"Summarize the following exchanges:\n{combined_text}"
            try:
                cluster_summary = llm_summary_fn(summary_prompt).strip()
            except Exception as e:
                cluster_summary = f"(Summary error: {e})"

            start_ts = dt.utcnow().isoformat()
            try:
                cur.execute(
                    """
                    INSERT INTO sleep_history (cycle_start, cycle_end, original_count, summary_text, notes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (start_ts, dt.utcnow().isoformat(), len(exp_list), cluster_summary, "")
                )
                conn.commit()
            except Exception:
                conn.rollback()

            summary_text_for_history += cluster_summary + "\n"

        message = f"Sleep Cycle: created {self.sleep_n_clusters} clusters and saved summaries."
        latest = {"original_count": sum(len(v) for v in cluster_map.values()),
                  "summary_text": summary_text_for_history.strip()}

        if rebuild_index:
            try:
                cur.execute("SELECT id, content FROM experiences")
                all_rows = cur.fetchall()
                if all_rows:
                    dim2 = self.embedder.encode(all_rows[0][1]).astype("float32").shape[0]
                    new_idx = faiss.IndexIDMap(faiss.IndexFlatL2(dim2))
                    for rec_id, content in all_rows:
                        vec2 = self.embedder.encode(content).astype("float32")
                        new_idx.add_with_ids(np.array([vec2]), np.array([int(rec_id)], dtype="int64"))
                    self.index = new_idx
                    faiss.write_index(self.index, self.index_file)
            except Exception:
                pass

        return {"message": message, "latest_sleep_history": latest}

    def handle_message(self, user_utterance: str, context_messages: list) -> str:
        """
        Single entry point for external apps.
        1. Automatically add the user utterance to memory.
        2. Run recall (including Sleep or NoRecall).
        """
        # Auto-add to memory
        title = user_utterance[:50]
        content = user_utterance
        summary_text = user_utterance
        try:
            self.add_experience(title=title, content=content, summary_text=summary_text)
        except Exception as e:
            print(f"[SynapseMemory] add_experience error: {e}")

        # Recall / Sleep / NoRecall flow
        try:
            return self.recall(query=user_utterance, context_messages=context_messages)
        except Exception as e:
            # Fallback: use context only
            fallback_prompt = f"Recent conversation:\n{''.join(context_messages)}\nUser question: {user_utterance}\nAnswer:"
            return self._llm_callback(fallback_prompt)

    def set_llm_callback(self, func):
        """Register an external LLM callback function."""
        self._llm_callback = func

    def close(self):
        """Close any resources."""
        conn = getattr(self._thread_local, "conn", None)
        if conn:
            conn.close()
