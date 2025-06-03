import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# 環境変数からAPIキーを読み込むか、直接設定
# API_KEY = os.getenv("YOUR_LLM_API_KEY") # 実際のAPIキーに置き換える

class Experience:
    """記憶の最小単位となる経験クラス"""
    def __init__(self, id: Optional[int], timestamp: str, content: str,
                 source_type: str, metadata: Dict, importance: float, processed: bool):
        self.id = id
        self.timestamp = timestamp
        self.content = content
        self.source_type = source_type
        self.metadata = metadata if isinstance(metadata, dict) else json.loads(metadata)
        self.importance = importance
        self.processed = processed

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "content": self.content,
            "source_type": self.source_type,
            "metadata": json.dumps(self.metadata),
            "importance": self.importance,
            "processed": self.processed
        }

class MemoryNode:
    """原子ノード（記憶から抽出された重要な概念や事実）"""
    def __init__(self, id: Optional[int], experience_id: int, text: str,
                 node_type: str, importance: float, access_count: int, last_accessed: str):
        self.id = id
        self.experience_id = experience_id
        self.text = text
        self.node_type = node_type
        self.importance = importance
        self.access_count = access_count
        self.last_accessed = last_accessed

    def to_dict(self):
        return {
            "id": self.id,
            "experience_id": self.experience_id,
            "text": self.text,
            "node_type": self.node_type,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }

class SynapseMemory:
    """
    Synapse記憶システムの中核クラス。
    長期記憶（SQLite）と短期記憶（ChromaDB）を管理。
    """
    def __init__(self, db_path: str = "synapse_memory.db",
                 chroma_path: str = "synapse_chroma_db",
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 debug: bool = False):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.embedding_model_name = embedding_model_name
        self.debug = debug
        self.connection = None
        self.chroma_client = None
        self.embedding_function = None
        self.chroma_collection = None

        self._initialize_components()
        self.init_db() # コンストラクタでDB初期化を呼び出す

    def _initialize_components(self):
        """EmbeddingモデルとChromaDBクライアントの初期化"""
        if self.debug:
            print(f"Initializing embedding model: {self.embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            if self.debug:
                print("Embedding model loaded.")
        except Exception as e:
            if self.debug:
                print(f"Error loading embedding model: {e}")
            raise # モデルロードに失敗したら例外を再スロー

        if self.debug:
            print(f"Initializing ChromaDB client at: {self.chroma_path}")
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="experiences_collection",
                embedding_function=self.embedding_function # ここでembedding_functionを渡す
            )
            if self.debug:
                print(f"ChromaDB initialized at: {self.chroma_path}")
        except Exception as e:
            if self.debug:
                print(f"Error initializing ChromaDB: {e}")
            raise # ChromaDB初期化に失敗したら例外を再スロー

    def init_db(self):
        """SQLiteデータベースの初期化と接続確立"""
        if self.connection:
            self.close_db() # 既存の接続があれば閉じる

        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row # 列名をキーとしてアクセスできるようにする
            if self.debug:
                print(f"SQLite database connected at: {self.db_path}")
            self._initialize_tables() # テーブルの初期化を呼び出す
        except Exception as e:
            if self.debug:
                print(f"Error connecting to SQLite database: {e}")
            raise # DB接続に失敗したら例外を再スロー


    def _initialize_tables(self):
        """SQLiteデータベースのテーブルスキーマ定義と作成"""
        cursor = self.connection.cursor()

        # experiences テーブルの作成
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL,
            source_type TEXT DEFAULT 'chat',
            metadata TEXT DEFAULT '{}',
            importance REAL DEFAULT 0.5, -- ★ この行が重要
            processed BOOLEAN DEFAULT FALSE
        )
        """)

        # memory_nodes テーブルの作成
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experience_id INTEGER,
            text TEXT NOT NULL,
            node_type TEXT DEFAULT 'statement',
            importance REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(experience_id) REFERENCES experiences(id)
        )
        """)

        # relationships テーブルの作成
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
            UNIQUE(source_node_id, target_node_id, relationship_type)
        )
        """)
        
        # sleep_cycles テーブルの作成
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sleep_cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_start DATETIME DEFAULT CURRENT_TIMESTAMP,
            cycle_end DATETIME,
            processed_experiences INTEGER,
            discovered_relationships INTEGER,
            notes TEXT
        )
        """)

        self.connection.commit()

        if self.debug:
            print("SQLite database tables initialized successfully.")

    def close_db(self):
        """データベース接続を閉じる"""
        if self.connection:
            self.connection.close()
            self.connection = None
            if self.debug:
                print("SQLite database connection closed.")

    def add_experience(self, content: str, source_type: str = 'chat',
                       metadata: Optional[Dict] = None, importance: float = 0.5) -> Experience:
        """新しい経験をデータベースに追加し、ChromaDBにも埋め込みを保存"""
        if metadata is None:
            metadata = {}
        
        # SQLiteへの追加
        cursor = self.connection.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO experiences (timestamp, content, source_type, metadata, importance) VALUES (?, ?, ?, ?, ?)",
            (timestamp, content, source_type, json.dumps(metadata), importance)
        )
        experience_id = cursor.lastrowid
        self.connection.commit()
        
        # ChromaDBへの追加
        try:
            # EmbeddingFunctionがセットされているか確認
            if self.chroma_collection and self.embedding_function:
                # contentからembeddingを直接生成
                embedding = self.embedding_model.encode([content]).tolist()[0]
                self.chroma_collection.add(
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{"source_type": source_type, "experience_id": experience_id}],
                    ids=[str(experience_id)] # experience_idをIDとして使用
                )
                if self.debug:
                    print(f"Added experience {experience_id} to ChromaDB.")
            else:
                if self.debug:
                    print("ChromaDB collection or embedding function not initialized. Skipping ChromaDB add.")
        except Exception as e:
            if self.debug:
                print(f"Error adding experience to ChromaDB: {e}")
            # エラーが発生してもSQLiteへの追加は成功しているため、例外を再スローしない

        return Experience(experience_id, timestamp, content, source_type, metadata, importance, False)

    def get_experience(self, experience_id: int) -> Optional[Experience]:
        """IDに基づいて経験を取得"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM experiences WHERE id = ?", (experience_id,))
        row = cursor.fetchone()
        if row:
            return Experience(**row)
        return None

    def recall_relevant_experiences(self, query: str, top_k: int = 5) -> List[Experience]:
        """クエリに基づいて関連性の高い経験を検索"""
        if not self.chroma_collection:
            if self.debug:
                print("ChromaDB collection not initialized. Cannot recall experiences.")
            return [] # ChromaDBが初期化されていない場合は空リストを返す

        try:
            # ChromaDBで類似検索
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )

            relevant_experiences = []
            if results and results['ids'] and results['ids'][0]:
                experience_ids = [int(id_str) for id_str in results['ids'][0]]
                # SQLiteから詳細情報を取得
                placeholders = ','.join('?' * len(experience_ids))
                cursor = self.connection.cursor()
                cursor.execute(f"SELECT * FROM experiences WHERE id IN ({placeholders})", experience_ids)
                
                # 検索結果の順序をChromaDBの結果に合わせる
                experiences_map = {exp.id: Experience(**exp) for exp in cursor.fetchall()}
                for exp_id in experience_ids:
                    if exp_id in experiences_map:
                        relevant_experiences.append(experiences_map[exp_id])
            
            if self.debug:
                print(f"Recalled {len(relevant_experiences)} experiences for query: '{query}'")
            return relevant_experiences
        except Exception as e:
            if self.debug:
                print(f"Error recalling relevant experiences: {e}")
            return []

    def get_unprocessed_experiences(self) -> List[Experience]:
        """未処理の経験を取得"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM experiences WHERE processed = FALSE ORDER BY timestamp ASC")
        return [Experience(**row) for row in cursor.fetchall()]

    def mark_experience_processed(self, experience_id: int):
        """経験を処理済みとしてマーク"""
        cursor = self.connection.cursor()
        cursor.execute("UPDATE experiences SET processed = TRUE WHERE id = ?", (experience_id,))
        self.connection.commit()
        if self.debug:
            print(f"Marked experience {experience_id} as processed.")

    def add_atomic_node(self, experience_id: int, text: str, node_type: str = 'statement',
                       importance: float = 0.5) -> MemoryNode:
        """原子ノードをデータベースに追加"""
        cursor = self.connection.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO memory_nodes (experience_id, text, node_type, importance, last_accessed) VALUES (?, ?, ?, ?, ?)",
            (experience_id, text, node_type, importance, timestamp)
        )
        node_id = cursor.lastrowid
        self.connection.commit()
        if self.debug:
            print(f"Added atomic node {node_id}: '{text[:30]}...'")
        return MemoryNode(node_id, experience_id, text, node_type, importance, 0, timestamp)

    def update_node_access(self, node_id: int):
        """原子ノードのアクセス回数と最終アクセス日時を更新"""
        cursor = self.connection.cursor()
        cursor.execute(
            "UPDATE memory_nodes SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (datetime.now().isoformat(), node_id)
        )
        self.connection.commit()
        if self.debug:
            print(f"Updated access for node {node_id}.")

    def add_relationship(self, source_node_id: int, target_node_id: int,
                         relationship_type: str = 'semantic', strength: float = 0.5,
                         discovered_method: str = 'auto') -> int:
        """ノード間の関係を追加"""
        cursor = self.connection.cursor()
        created_at = datetime.now().isoformat()
        try:
            cursor.execute(
                "INSERT INTO relationships (source_node_id, target_node_id, relationship_type, strength, discovered_method, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (source_node_id, target_node_id, relationship_type, strength, discovered_method, created_at)
            )
            relationship_id = cursor.lastrowid
            self.connection.commit()
            if self.debug:
                print(f"Added relationship {relationship_id} between {source_node_id} and {target_node_id}.")
            return relationship_id
        except sqlite3.IntegrityError:
            if self.debug:
                print(f"Relationship already exists between {source_node_id} and {target_node_id} with type {relationship_type}. Skipping.")
            return -1 # 既に存在する関係は追加しない
        except Exception as e:
            if self.debug:
                print(f"Error adding relationship: {e}")
            raise

    def get_related_nodes(self, node_id: int, relationship_type: Optional[str] = None, limit: int = 10) -> List[MemoryNode]:
        """特定のノードに関連するノードを取得"""
        cursor = self.connection.cursor()
        query = """
            SELECT mn.* FROM memory_nodes mn
            JOIN relationships r ON mn.id = r.target_node_id
            WHERE r.source_node_id = ?
        """
        params = [node_id]
        if relationship_type:
            query += " AND r.relationship_type = ?"
            params.append(relationship_type)
        query += " ORDER BY r.strength DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, tuple(params))
        return [MemoryNode(**row) for row in cursor.fetchall()]

    def sleep_cycle(self) -> str:
        """
        AIの「スリープサイクル」プロセス。
        未処理の経験を処理し、原子ノードを生成し、関連を構築する。
        """
        if self.debug:
            print("Starting sleep cycle...")
        cycle_start = datetime.now().isoformat()
        
        unprocessed_experiences = self.get_unprocessed_experiences()
        processed_count = 0
        discovered_relationships = 0

        if not unprocessed_experiences:
            report = "処理すべき新しい経験はありませんでした。"
            self._record_sleep_cycle(cycle_start, datetime.now().isoformat(), 0, 0, report)
            return report

        for experience in unprocessed_experiences:
            if self.debug:
                print(f"Processing experience {experience.id}: {experience.content[:50]}...")
            
            # 1. 経験から原子ノードを抽出（ダミー実装）
            # 実際にはLLMを使ってテキストを解析し、重要な事実や概念を抽出する
            nodes_text = self._extract_atomic_nodes_from_experience(experience.content)
            
            for node_text in nodes_text:
                atomic_node = self.add_atomic_node(experience.id, node_text)
                
                # 2. 既存のノードとの関連を構築（ダミー実装）
                # 実際には、新しく生成したノードと、既存の関連性の高いノードを探し、
                # LLMを使って関係性を推論し、strengthを計算する
                existing_nodes = self._get_most_recent_nodes(limit=5) # 例として最近のノードをいくつか取得
                for existing_node in existing_nodes:
                    if existing_node.id != atomic_node.id:
                        # ダミーで関係性を追加
                        self.add_relationship(atomic_node.id, existing_node.id, strength=np.random.rand())
                        discovered_relationships += 1
                        self.add_relationship(existing_node.id, atomic_node.id, strength=np.random.rand()) # 双方向
                        discovered_relationships += 1
            
            self.mark_experience_processed(experience.id)
            processed_count += 1
            
        cycle_end = datetime.now().isoformat()
        report = f"スリープサイクル完了。\n処理済み経験数: {processed_count}\n発見された関係数: {discovered_relationships}"
        self._record_sleep_cycle(cycle_start, cycle_end, processed_count, discovered_relationships, report)
        
        if self.debug:
            print(f"Sleep cycle finished. Report: {report}")
        return report
    
    def _record_sleep_cycle(self, cycle_start: str, cycle_end: str,
                           processed_experiences: int, discovered_relationships: int, notes: str):
        """スリープサイクルの結果を記録"""
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO sleep_cycles (cycle_start, cycle_end, processed_experiences, discovered_relationships, notes) VALUES (?, ?, ?, ?, ?)",
            (cycle_start, cycle_end, processed_experiences, discovered_relationships, notes)
        )
        self.connection.commit()
        if self.debug:
            print("Sleep cycle recorded.")


    def _extract_atomic_nodes_from_experience(self, content: str) -> List[str]:
        """
        経験コンテンツから原子ノードを抽出するダミー関数。
        実際にはLLMを使用する。
        """
        # 簡単な例として、文の区切りで分割
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        return sentences
    
    def _get_most_recent_nodes(self, limit: int = 5) -> List[MemoryNode]:
        """最近作成されたノードを取得するダミー関数"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM memory_nodes ORDER BY last_accessed DESC LIMIT ?", (limit,))
        return [MemoryNode(**row) for row in cursor.fetchall()]

# --- 以下は synapse_studio.py との重複を避けるための参考情報 ---
# このファイルはsynapse_memoryパッケージとしてpip installされることを想定。
# そのため、__main__ブロックは通常含まれない。
# if __name__ == '__main__':
#     # テストコード
#     mem = SynapseMemory(debug=True)
#     exp1 = mem.add_experience("ユーザーがPythonでGUIアプリを開発したいと述べた。", "chat")
#     exp2 = mem.add_experience("PyQt5とQTextEditを使ってエディタを作る。", "thought")
    
#     time.sleep(1) # 時間差をつける
#     mem.sleep_cycle()
    
#     relevant_exp = mem.recall_relevant_experiences("GUI開発について")
#     print("\n関連経験:")
#     for exp in relevant_exp:
#         print(f"- {exp.content}")
        
#     mem.close_db()
