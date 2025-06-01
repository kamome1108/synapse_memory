import sqlite3
import json
import time
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List, Optional
import chromadb
from datetime import datetime

class Experience:
    """Represents a raw user or system experience."""
    def __init__(self, content: str, source_type: str = 'chat', metadata: Optional[Dict] = None):
        self.content = content
        self.source_type = source_type
        self.metadata = json.dumps(metadata) if metadata else '{}'
        self.hash_id = hashlib.sha256(content.encode()).hexdigest()

class MemoryNode:
    """Represents an atomic piece of memory derived from an experience."""
    def __init__(self, text: str, node_type: str = 'statement', importance: float = 0.5):
        self.text = text
        self.node_type = node_type
        self.importance = importance

class SynapseMemory:
    """
    Experience-based memory system inspired by neural synapses.
    Designed for flexible development and future API integration.
    Utilizes SQLite for metadata/relationships and ChromaDB for vector search.
    """

    def __init__(self, db_path: str = "synapse_memory.db", chroma_path: str = "synapse_chroma_db", debug: bool = False):
        self.db_path = db_path
        self.debug = debug
        self.connection = None
        
        # ChromaDBの初期化
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="memory_nodes_collection",
                # SynapseMemoryで埋め込みを生成するので、ChromaDB側ではembedding_functionを指定しない
            )
            if self.debug:
                print(f"ChromaDB initialized at: {chroma_path}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}. Please check permissions or installation.")
            raise

        self._initialize_database()

        # Sentence Transformer モデルのロード
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            if self.debug:
                print("Embedding model 'all-MiniLM-L6-v2' loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}. Please ensure you have an internet connection for the first run.")
            raise

        # Development-friendly configuration
        self.config = {
            "max_recall_results": 10,
            "importance_threshold": 0.3,
            "sleep_batch_size": 20,
            "similarity_threshold": 0.5, # コサイン類似度の閾値
            "debug_mode": debug
        }

        if self.debug:
            print(f"SynapseMemory initialized with database: {db_path}")

    def _initialize_database(self):
        """Initialize SQLite database with flexible schema"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row # カラム名をキーとしてアクセスできるようにする

        cursor = self.connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source_type TEXT DEFAULT 'chat',
            metadata TEXT DEFAULT '{}',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            hash_id TEXT UNIQUE,
            processed BOOLEAN DEFAULT FALSE
        )
        """)

        # embedding_vector カラムは不要になった (ChromaDBが管理するため)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_nodes (
            id INTEGER PRIMARY PRIMARY KEY AUTOINCREMENT,
            experience_id INTEGER,
            text TEXT NOT NULL,
            node_type TEXT DEFAULT 'statement',
            importance REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(experience_id) REFERENCES experiences(id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_node_id INTEGER,
            target_node_id INTEGER,
            relationship_type TEXT DEFAULT 'semantic',
            strength REAL DEFAULT 0.5,
            discovered_method TEXT DEFAULT 'auto',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_node_id) REFERENCES memory_nodes(id),
            FOREIGN KEY(target_node_id) REFERENCES memory_nodes(id),
            UNIQUE(source_node_id, target_node_id, relationship_type) -- 同じ関係性を重複して追加しない
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sleep_cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_start DATETIME,
            cycle_end DATETIME,
            processed_experiences INTEGER,
            discovered_relationships INTEGER,
            notes TEXT
        )
        """)

        self.connection.commit()

        if self.debug:
            print("SQLite database initialized successfully")

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            if self.debug:
                print("Database connection closed.")

    def add_experience(self, content: str, source_type: str = 'chat', metadata: Optional[Dict] = None) -> Optional[int]:
        """
        Adds a new experience to the memory.
        Returns the ID of the new experience or None if it's a duplicate.
        """
        experience = Experience(content, source_type, metadata)
        cursor = self.connection.cursor()
        
        try:
            cursor.execute("SELECT id FROM experiences WHERE hash_id = ?", (experience.hash_id,))
            if cursor.fetchone():
                if self.debug:
                    print(f"Duplicate experience detected, skipping: {content[:50]}...")
                return None # Duplicate experience

            cursor.execute("""
            INSERT INTO experiences (content, source_type, metadata, hash_id)
            VALUES (?, ?, ?, ?)
            """, (experience.content, experience.source_type, experience.metadata, experience.hash_id))
            self.connection.commit()
            exp_id = cursor.lastrowid
            if self.debug:
                print(f"Experience added: ID={exp_id}, Type={source_type}")
            return exp_id
        except sqlite3.IntegrityError: # Hash ID unique constraint violated in rare race condition
            if self.debug:
                print(f"IntegrityError: Likely race condition with duplicate hash_id for {content[:50]}...")
            return None
        except Exception as e:
            if self.debug:
                print(f"Error adding experience: {e}")
            self.connection.rollback()
            return None

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting logic."""
        # This is a very basic splitter. For production, consider NLTK or SpaCy.
        sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if s.strip()]
        return sentences

    def _classify_sentence(self, sentence: str) -> str:
        """Classify sentence type (e.g., statement, question, task, code, tool)"""
        lower_s = sentence.lower()
        if '?' in sentence:
            return 'question'
        if any(keyword in lower_s for keyword in ['task:', 'todo:', 'implement', 'fix', 'develop']):
            return 'task'
        if any(keyword in lower_s for keyword in ['def ', 'import ', 'class ', 'return ', 'print(']): # Python-specific
            return 'code'
        if any(keyword in lower_s for keyword in ['tool:', 'use ', 'execute ']):
            return 'tool'
        return 'statement'

    def _calculate_importance(self, sentence: str) -> float:
        """
        A placeholder for calculating sentence importance.
        Can be expanded with LLM-based assessment.
        """
        length_score = min(len(sentence) / 100.0, 1.0) # Longer sentences might be more important
        keyword_score = 0
        if any(k in sentence.lower() for k in ['error', 'important', 'critical', 'solution', 'goal']):
            keyword_score = 0.2
        return 0.5 + (length_score * 0.3) + keyword_score # Base importance 0.5

    def _process_experience_to_nodes(self, experience_id: int, content: str):
        """Process experience into atomic memory nodes and add to SQLite and ChromaDB."""
        sentences = self._split_into_sentences(content)
        cursor = self.connection.cursor()
        
        node_ids_to_add_to_chroma = []
        documents_to_add_to_chroma = []
        embeddings_to_add_to_chroma = []
        metadatas_to_add_to_chroma = []

        for sentence in sentences:
            if len(sentence.strip()) < 10: # 短すぎる文はスキップ
                continue

            node_type = self._classify_sentence(sentence)
            importance = self._calculate_importance(sentence)
            
            # SQLiteにまず追加し、IDを取得
            cursor.execute("""
            INSERT INTO memory_nodes (experience_id, text, node_type, importance)
            VALUES (?, ?, ?, ?)
            """, (experience_id, sentence, node_type, importance))
            node_id = cursor.lastrowid # SQLiteで生成されたID

            # ChromaDBに追加するためのデータを準備
            embedding = self.embedding_model.encode(sentence).tolist() # リスト形式で渡す
            node_ids_to_add_to_chroma.append(str(node_id))
            documents_to_add_to_chroma.append(sentence)
            embeddings_to_add_to_chroma.append(embedding)
            metadatas_to_add_to_chroma.append({
                "experience_id": experience_id,
                "node_type": node_type,
                "importance": importance # メタデータとしても重要度を保存
            })
        
        # バッチでChromaDBに追加
        if node_ids_to_add_to_chroma:
            try:
                self.chroma_collection.add(
                    documents=documents_to_add_to_chroma,
                    embeddings=embeddings_to_add_to_chroma,
                    metadatas=metadatas_to_add_to_chroma,
                    ids=node_ids_to_add_to_chroma
                )
                if self.debug:
                    print(f"Added {len(node_ids_to_add_to_chroma)} nodes to ChromaDB for experience {experience_id}")
            except Exception as e:
                if self.debug:
                    print(f"Error adding to ChromaDB for experience {experience_id}: {e}")
                # ChromaDBへの追加が失敗しても、SQLiteのコミットは行われるようにする（堅牢性のため）
        
        self.connection.commit() # SQLiteのコミット

    def recall_memory(self, query: str, context: Optional[Dict[str, Any]] = None,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Recall relevant memories based on query using embedding similarity via ChromaDB.

        Args:
            query: Search query
            context: Additional context for filtering (e.g., {"node_type": "code"})
            limit: Maximum number of results

        Returns:
            List of relevant memory nodes with metadata and similarity score
        """
        if limit is None:
            limit = self.config["max_recall_results"]

        chroma_where_clause = {}
        if context and "node_type" in context:
            chroma_where_clause["node_type"] = context["node_type"]
        
        # ChromaDBで類似度検索を実行
        try:
            # ChromaDBのクエリにwhere句を追加して、重要度閾値と類似度閾値をフィルタリングする
            # importanceはメタデータに保存されているので、where句で利用できる
            chroma_results = self.chroma_collection.query(
                query_texts=[query],
                n_results=limit * 2, # 少し多めに取得して、後で重要度でフィルタリングする余裕を持たせる
                where=chroma_where_clause,
                where_document={"$and": [ # ドキュメントの内容に基づくフィルタリング（例: 特定のキーワードを含む）
                    # {"$contains": "important"},
                ]}
            )
        except Exception as e:
            if self.debug:
                print(f"Error querying ChromaDB: {e}")
            return []

        recalled_node_ids = []
        distances_map = {}
        if chroma_results and chroma_results['ids'] and chroma_results['ids'][0]:
            for i, node_id_str in enumerate(chroma_results['ids'][0]):
                node_id = int(node_id_str)
                distance = chroma_results['distances'][0][i]
                # ChromaDBの距離は小さいほど似ているため、類似度 (0-1) に変換する
                similarity = 1 - distance # 0.0 (完全に一致) から 1.0 (全く似ていない) を反転
                
                # ここでChromaDBから取得したメタデータ（重要度など）を使ってフィルタリング
                # ただし、ChromaDBのwhere句でimportanceを使うことも可能
                # ここではSQLiteから改めて取得するため、単純にIDを収集
                recalled_node_ids.append(node_id)
                distances_map[node_id] = similarity # 類似度をマップに保存

        if not recalled_node_ids:
            if self.debug:
                print(f"No initial results from ChromaDB for query: '{query}'")
            return []

        # SQLiteから詳細情報を取得し、重要度フィルタリングを適用
        placeholders = ','.join('?' * len(recalled_node_ids))
        cursor = self.connection.cursor()
        cursor.execute(f"""
        SELECT mn.id, mn.text, mn.node_type, mn.importance,
               e.source_type, e.metadata, e.timestamp, mn.access_count
        FROM memory_nodes mn
        JOIN experiences e ON mn.experience_id = e.id
        WHERE mn.id IN ({placeholders}) AND mn.importance >= ?
        """, recalled_node_ids + [self.config["importance_threshold"]])

        final_results = []
        for row in cursor.fetchall():
            result = dict(row)
            # ChromaDBで計算した類似度を付与
            result['similarity'] = distances_map.get(result['id'], 0.0)
            
            # 類似度閾値でフィルタリング（ChromaDBの`where`句でも可能だが、ここでは明確化のためPythonで再度）
            if result['similarity'] >= self.config["similarity_threshold"]:
                final_results.append(result)
                self._update_access_pattern(result['id']) # アクセスパターンを更新

        # 最終的に類似度でソートし、上限を適用
        final_results.sort(key=lambda x: x['similarity'], reverse=True)
        results_to_return = final_results[:limit]

        if self.debug:
            print(f"Recalled {len(results_to_return)} memories for query: '{query}' (using ChromaDB and SQLite)")

        return results_to_return

    def _update_access_pattern(self, node_id: int):
        """Update access count and last accessed timestamp for a memory node."""
        cursor = self.connection.cursor()
        cursor.execute("""
        UPDATE memory_nodes
        SET access_count = access_count + 1,
            last_accessed = CURRENT_TIMESTAMP
        WHERE id = ?
        """, (node_id,))
        self.connection.commit()

    def _add_relationship(self, source_node_id: int, target_node_id: int,
                          relationship_type: str = 'semantic', strength: float = 0.5,
                          discovered_method: str = 'auto'):
        """Adds a relationship between two memory nodes."""
        if source_node_id == target_node_id: # 自分自身への関係は追加しない
            return
            
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
            INSERT INTO relationships (source_node_id, target_node_id, relationship_type, strength, discovered_method)
            VALUES (?, ?, ?, ?, ?)
            """, (source_node_id, target_node_id, relationship_type, strength, discovered_method))
            self.connection.commit()
            if self.debug:
                print(f"Relationship added: {source_node_id} --({relationship_type} s={strength})--> {target_node_id}")
        except sqlite3.IntegrityError:
            if self.debug:
                print(f"Relationship already exists or invalid: {source_node_id}-{target_node_id}-{relationship_type}")
        except Exception as e:
            if self.debug:
                print(f"Error adding relationship: {e}")
            self.connection.rollback()


    def _find_similar_nodes(self, text: str, exclude_experience: Optional[int] = None, limit: int = 5) -> List[Dict]:
        """Find similar nodes using ChromaDB for sleep processing, excluding a specific experience if provided."""
        
        chroma_where_clause = {}
        if exclude_experience is not None:
            chroma_where_clause["experience_id"] = {"$ne": exclude_experience}

        try:
            chroma_results = self.chroma_collection.query(
                query_texts=[text],
                n_results=limit * 2, # 少し多めに取得して、後でフィルタリングする
                where=chroma_where_clause,
                # include=['documents', 'distances', 'metadatas', 'ids'] # デフォルトで含まれる
            )
        except Exception as e:
            if self.debug:
                print(f"Error querying ChromaDB in _find_similar_nodes: {e}")
            return []

        similar_node_ids = []
        distances_map = {}
        if chroma_results and chroma_results['ids'] and chroma_results['ids'][0]:
            for i, node_id_str in enumerate(chroma_results['ids'][0]):
                node_id = int(node_id_str)
                distance = chroma_results['distances'][0][i]
                similarity = 1 - distance 
                if similarity >= self.config["similarity_threshold"]: # 類似度閾値でフィルタ
                    similar_node_ids.append(node_id)
                    distances_map[node_id] = similarity

        if not similar_node_ids:
            return []

        # SQLiteから詳細情報を取得
        placeholders = ','.join('?' * len(similar_node_ids))
        cursor = self.connection.cursor()
        cursor.execute(f"""
        SELECT id, text, node_type, importance
        FROM memory_nodes
        WHERE id IN ({placeholders}) AND importance >= ?
        """, similar_node_ids + [self.config["importance_threshold"]])

        final_similar_nodes = []
        for row in cursor.fetchall():
            node_data = dict(row)
            node_data['similarity'] = distances_map.get(node_data['id'], 0.0) # 類似度を付与
            final_similar_nodes.append(node_data)
        
        final_similar_nodes.sort(key=lambda x: x['similarity'], reverse=True)

        return final_similar_nodes[:limit]


    def sleep_process(self) -> Dict[str, Any]:
        """
        Performs background processing (like synaptic consolidation).
        Discovers new relationships and updates importance.
        """
        if self.debug:
            print("\nStarting sleep process...")
        
        cursor = self.connection.cursor()
        start_time = datetime.now()
        processed_exp_count = 0
        discovered_rel_count = 0

        # 未処理の経験を取得
        cursor.execute("""
        SELECT id, content, timestamp, metadata
        FROM experiences
        WHERE processed = FALSE
        ORDER BY timestamp ASC
        LIMIT ?
        """, (self.config["sleep_batch_size"],))
        unprocessed_experiences = cursor.fetchall()

        if not unprocessed_experiences:
            if self.debug:
                print("No unprocessed experiences found. Sleep process skipped.")
            return {
                "processed_experiences": 0,
                "discovered_relationships": 0,
                "notes": "No unprocessed experiences."
            }

        for exp_row in unprocessed_experiences:
            experience_id = exp_row['id']
            experience_content = exp_row['content']
            experience_timestamp = datetime.strptime(exp_row['timestamp'], '%Y-%m-%d %H:%M:%S.%f') if '.' in exp_row['timestamp'] else datetime.strptime(exp_row['timestamp'], '%Y-%m-%d %H:%M:%S')
            experience_metadata = json.loads(exp_row['metadata'])

            # 経験からメモリノードを生成
            self._process_experience_to_nodes(experience_id, experience_content)
            processed_exp_count += 1
            
            # --- 関係性発見のロジックを強化 ---

            # 1. 時間的近接性に基づく関係性 (新しいノードと直前のノード)
            # 現在の経験のノードを取得
            cursor.execute("SELECT id, text, node_type FROM memory_nodes WHERE experience_id = ?", (experience_id,))
            current_exp_nodes = cursor.fetchall()
            
            # 直前の経験から生成されたノードを取得 (単純な時間的近接性)
            cursor.execute("""
            SELECT mn.id, mn.text, mn.node_type
            FROM memory_nodes mn
            JOIN experiences e ON mn.experience_id = e.id
            WHERE e.timestamp < ? AND mn.node_type != 'tool' -- ツールは頻繁に呼ばれるので除外
            ORDER BY e.timestamp DESC, mn.id DESC
            LIMIT 5 -- 直前の数個のノード
            """, (experience_timestamp,))
            previous_nodes = cursor.fetchall()

            for current_node in current_exp_nodes:
                for prev_node in previous_nodes:
                    # 同じ経験内のノード同士、または自身への関係はスキップ
                    if current_node['id'] == prev_node['id'] or current_node['experience_id'] == prev_node['experience_id']:
                        continue
                    
                    # 例: 質問の後に回答があれば関係性
                    if prev_node['node_type'] == 'question' and current_node['node_type'] == 'statement':
                        self._add_relationship(prev_node['id'], current_node['id'], 'answers', strength=0.8, discovered_method='temporal+type')
                        discovered_rel_count += 1
                    
                    # 同一ソースタイプでの連続性
                    # if experience_metadata.get('source_type') == json.loads(self._get_experience_metadata(prev_node['experience_id'])).get('source_type'):
                    #     self._add_relationship(prev_node['id'], current_node['id'], 'temporal_flow', strength=0.6, discovered_method='temporal+source')
                    #     discovered_rel_count += 1
                    
                    # その他の時間的関係性 (デフォルト)
                    self._add_relationship(prev_node['id'], current_node['id'], 'temporal_follows', strength=0.4, discovered_method='temporal')
                    discovered_rel_count += 1
            
            # 2. 意味的類似性に基づく関係性 (既存の全てのノードから検索)
            for current_node in current_exp_nodes:
                # 自身が属する経験のノードは除外して、類似ノードを検索
                similar_nodes = self._find_similar_nodes(current_node['text'], exclude_experience=experience_id, limit=3)
                for sim_node in similar_nodes:
                    # 類似度を strength に変換 (0.5以上を考慮)
                    strength = sim_node['similarity'] 
                    if strength >= self.config["similarity_threshold"]:
                         self._add_relationship(current_node['id'], sim_node['id'], 'semantic_similarity', strength=strength, discovered_method='embedding')
                         discovered_rel_count += 1
                         # 双方向の関係も追加 (オプション)
                         # self._add_relationship(sim_node['id'], current_node['id'], 'semantic_similarity', strength=strength, discovered_method='embedding_reciprocal')


            # 経験を処理済みとしてマーク
            cursor.execute("UPDATE experiences SET processed = TRUE WHERE id = ?", (experience_id,))
            self.connection.commit()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        notes = f"Processed {processed_exp_count} experiences, discovered {discovered_rel_count} relationships in {duration:.2f} seconds."

        cursor.execute("""
        INSERT INTO sleep_cycles (cycle_start, cycle_end, processed_experiences, discovered_relationships, notes)
        VALUES (?, ?, ?, ?, ?)
        """, (start_time.strftime('%Y-%m-%d %H:%M:%S.%f'), end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
              processed_exp_count, discovered_rel_count, notes))
        self.connection.commit()

        if self.debug:
            print(f"Sleep process completed. {notes}")
        
        return {
            "processed_experiences": processed_exp_count,
            "discovered_relationships": discovered_rel_count,
            "notes": notes
        }

    def get_stats(self) -> Dict[str, int]:
        """Returns basic statistics of the memory system."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiences")
        total_experiences = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM memory_nodes")
        total_memory_nodes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM relationships")
        total_relationships = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sleep_cycles")
        sleep_cycles_completed = cursor.fetchone()[0]

        return {
            "total_experiences": total_experiences,
            "total_memory_nodes": total_memory_nodes,
            "total_relationships": total_relationships,
            "sleep_cycles_completed": sleep_cycles_completed
        }

    def get_related_memories(self, node_id: int, relationship_type: Optional[str] = None,
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves memories related to a given node by following relationships.
        """
        cursor = self.connection.cursor()
        query = """
        SELECT mn.id, mn.text, mn.node_type, mn.importance,
               r.relationship_type, r.strength, r.discovered_method
        FROM relationships r
        JOIN memory_nodes mn ON r.target_node_id = mn.id
        WHERE r.source_node_id = ?
        """
        params = [node_id]

        if relationship_type:
            query += " AND r.relationship_type = ?"
            params.append(relationship_type)
        
        query += " ORDER BY r.strength DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
        
        if self.debug:
            print(f"Found {len(results)} related memories for node {node_id}")
        return results

    def get_node_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single memory node by its ID."""
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT mn.id, mn.text, mn.node_type, mn.importance, mn.access_count, mn.last_accessed,
               e.source_type, e.metadata, e.timestamp as experience_timestamp
        FROM memory_nodes mn
        JOIN experiences e ON mn.experience_id = e.id
        WHERE mn.id = ?
        """, (node_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['metadata'] = json.loads(result['metadata']) # JSON文字列をPython辞書に戻す
            return result
        return None

    def get_experience_by_id(self, experience_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single experience by its ID."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM experiences WHERE id = ?", (experience_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['metadata'] = json.loads(result['metadata']) # JSON文字列をPython辞書に戻す
            return result
        return None

    def _get_experience_metadata(self, experience_id: int) -> Optional[str]:
        """Helper to get metadata of an experience by ID."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT metadata FROM experiences WHERE id = ?", (experience_id,))
        row = cursor.fetchone()
        if row:
            return row['metadata']
        return None

    def get_project_context(self, project_name: str = None) -> Dict[str, Any]:
        """Get project-related context for development. Can be extended with embedding search."""
        # Note: This method still relies on text matching for project_name in metadata.
        # For a full embedding-based context recall, you would recall relevant nodes
        # based on project_name embedding against all memory node embeddings.
        
        cursor = self.connection.cursor()
        query_parts = [
            "SELECT mn.text, mn.node_type, e.source_type, e.timestamp",
            "FROM memory_nodes mn JOIN experiences e ON mn.experience_id = e.id",
            "WHERE 1=1"
        ]
        params = []

        if project_name:
            query_parts.append("AND e.metadata LIKE ?")
            params.append(f'%"{project_name}"%') # project_nameがJSON文字列の一部として存在することを想定
        
        # 開発関連のノードタイプを優先
        query_parts.append("AND mn.node_type IN ('code', 'task', 'tool', 'statement', 'question')")
        query_parts.append("ORDER BY e.timestamp DESC LIMIT 20")

        full_query = " ".join(query_parts)
        cursor.execute(full_query, params)
        
        results = cursor.fetchall()

        context = {
            'code_snippets': [],
            'tasks': [],
            'tools': [],
            'general_notes': [] # general から変更
        }

        for row in results:
            if row['node_type'] == 'code':
                context['code_snippets'].append(row['text'])
            elif row['node_type'] == 'task':
                context['tasks'].append(row['text'])
            elif row['node_type'] == 'tool':
                context['tools'].append(row['text'])
            else: # statement, question など
                context['general_notes'].append(row['text'])

        if self.debug:
            print(f"Retrieved project context for '{project_name if project_name else 'any'}'.")
        return context

# ====================================================================
# デモンストレーションコード（利用例）
# ====================================================================
if __name__ == "__main__":
    # 既存のDBファイルをクリーンアップして新しいスタートを切る場合
    import os
    if os.path.exists("synapse_memory.db"):
        os.remove("synapse_memory.db")
    if os.path.exists("synapse_chroma_db"): # ChromaDBのデータディレクトリも削除
        import shutil
        shutil.rmtree("synapse_chroma_db")
        
    print("--- Initializing SynapseMemory ---")
    memory = SynapseMemory(debug=True)

    # 1. 経験の追加
    print("\n--- Adding Experiences ---")
    exp1_id = memory.add_experience("Pythonでウェブアプリケーションを開発中です。バックエンドはFastAPIを使っています。", "chat", {"project": "web_app_project"})
    exp2_id = memory.add_experience("データベース接続にSQLiteを使っています。ORMはSQLAlchemyを採用しました。", "task", {"project": "web_app_project", "status": "completed"})
    exp3_id = memory.add_experience("フロントエンドはReactで構築し、TypeScriptも導入しています。", "chat", {"project": "web_app_project"})
    exp4_id = memory.add_experience("昨日、ユーザー認証機能の実装について話し合いました。OAuth2の導入を検討します。", "chat", {"project": "web_app_project"})
    exp5_id = memory.add_experience("テストコードを書き始める必要があります。pytestを使う予定です。", "task", {"project": "web_app_project", "status": "pending"})
    exp6_id = memory.add_experience("今日のランチは何を食べようか？カレーもいいな。", "chat", {"project": "daily_life"})
    exp7_id = memory.add_experience("SQLの結合について復習が必要です。", "statement", {"subject": "database"})
    exp8_id = memory.add_experience("ChromaDBの高速検索機能は素晴らしい。", "statement", {"subject": "ChromaDB"})
    
    print("\n--- Current Memory Stats ---")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # 2. スリーププロセス実行 (ノード生成と関係性発見)
    print("\n--- Running Sleep Process ---")
    sleep_results = memory.sleep_process()
    print(f"Sleep Process Notes: {sleep_results['notes']}")

    print("\n--- Stats after Sleep Process ---")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # 3. 記憶の想起 (埋め込み類似度検索)
    print("\n--- Recalling Memories ---")
    query = "Web開発のプロジェクトについて教えてください。"
    recalled_mems = memory.recall_memory(query, limit=5)
    print(f"\nQuery: '{query}'")
    for i, mem in enumerate(recalled_mems):
        print(f"{i+1}. [Similarity: {mem['similarity']:.4f}, Type: {mem['node_type']}, Source: {mem['source_type']}] {mem['text']}")

    query_code = "Pythonのテストコードについて"
    recalled_code_mems = memory.recall_memory(query_code, limit=3, context={"node_type": "task"})
    print(f"\nQuery (filtered by task): '{query_code}'")
    if recalled_code_mems:
        for i, mem in enumerate(recalled_code_mems):
            print(f"{i+1}. [Similarity: {mem['similarity']:.4f}, Type: {mem['node_type']}] {mem['text']}")
    else:
        print("No relevant task memories found.")

    query_unrelated = "今日の天気は？"
    recalled_unrelated = memory.recall_memory(query_unrelated, limit=1)
    print(f"\nQuery: '{query_unrelated}'")
    if recalled_unrelated:
        print(f"1. [Similarity: {recalled_unrelated[0]['similarity']:.4f}, Type: {recalled_unrelated[0]['node_type']}] {recalled_unrelated[0]['text']}")
    else:
        print("No relevant memories found for this query.")

    # 4. 関係性の取得
    print("\n--- Getting Related Memories for a Node ---")
    # 例: 最初のウェブアプリ関連のノードIDを取得
    cursor = memory.connection.cursor()
    cursor.execute("SELECT id, text FROM memory_nodes WHERE text LIKE '%FastAPI%' LIMIT 1")
    fastapi_node = cursor.fetchone()
    
    if fastapi_node:
        print(f"\nRelated memories for node ID {fastapi_node['id']} ('{fastapi_node['text']}'):")
        related_mems = memory.get_related_memories(fastapi_node['id'])
        if related_mems:
            for i, rel_mem in enumerate(related_mems):
                target_node_text = memory.get_node_by_id(rel_mem['id'])['text']
                print(f"{i+1}. [Relationship: {rel_mem['relationship_type']}, Strength: {rel_mem['strength']:.2f}] Target Node: {target_node_text}")
        else:
            print("No explicit relationships found.")
    else:
        print("FastAPI node not found for demonstration.")

    # 5. プロジェクトコンテキストの取得
    print("\n--- Getting Project Context ---")
    project_context = memory.get_project_context(project_name="web_app_project")
    print(f"\nContext for 'web_app_project':")
    for category, items in project_context.items():
        if items:
            print(f"  {category.replace('_', ' ').title()}:")
            for item in items:
                print(f"    - {item}")
        else:
            print(f"  {category.replace('_', ' ').title()}: None")
            
    daily_context = memory.get_project_context(project_name="daily_life")
    print(f"\nContext for 'daily_life': {daily_context.get('general_notes', [])}")

    # データベース接続を閉じる
    memory.close()
    print("\n--- SynapseMemory Demo Finished ---")