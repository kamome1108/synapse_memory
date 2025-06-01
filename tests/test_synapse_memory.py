import pytest
import os
import shutil
from synapse_memory.synapse_memory import SynapseMemory, Experience, MemoryNode

@pytest.fixture
def clean_db():
    db_path = "test_synapse.db"
    chroma_path = "test_synapse_chroma_db"

    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    synapse = SynapseMemory(db_path=db_path, chroma_path=chroma_path, debug=False)
    
    yield synapse
    
    synapse.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

def test_initialization(clean_db):
    synapse = clean_db
    stats = synapse.get_stats()
    assert stats['total_experiences'] == 0
    assert stats['total_memory_nodes'] == 0
    assert stats['total_relationships'] == 0

def test_add_experience(clean_db):
    synapse = clean_db
    exp_id = synapse.add_experience("This is a test experience.", "test_source")
    assert exp_id is not None
    stats = synapse.get_stats()
    assert stats['total_experiences'] == 1

def test_add_duplicate_experience(clean_db):
    synapse = clean_db
    content = "This is a duplicate experience."
    exp_id1 = synapse.add_experience(content, "test_source")
    exp_id2 = synapse.add_experience(content, "test_source")
    assert exp_id1 is not None
    assert exp_id2 is None
    stats = synapse.get_stats()
    assert stats['total_experiences'] == 1

def test_sleep_process_nodes_and_embeddings(clean_db):
    synapse = clean_db
    exp_id = synapse.add_experience("これはテスト用の文章です。いくつかの文に分割されるでしょう。", "test_source")
    
    sleep_results = synapse.sleep_process()
    assert sleep_results['processed_experiences'] == 1

    stats = synapse.get_stats()
    assert stats['total_memory_nodes'] > 0

    chroma_count = synapse.chroma_collection.count()
    assert chroma_count == stats['total_memory_nodes']


def test_recall_memory_embedding_similarity(clean_db):
    synapse = clean_db
    synapse.add_experience("Pythonでウェブアプリケーションを開発しています。", "chat")
    synapse.add_experience("データベース接続にSQLiteを使っています。", "tool")
    synapse.add_experience("フロントエンドはReactで構築し、Reduxも使います。", "chat")
    synapse.add_experience("今日の天気は晴れです。", "chat")

    synapse.sleep_process()

    results = synapse.recall_memory("Web開発について何か知ってる？", limit=3)
    assert len(results) > 0

    assert results[0]['similarity'] > results[-1]['similarity']

    related_content_found = False
    for res in results:
        if "ウェブアプリケーション" in res['text'] or "React" in res['text'] or "SQLite" in res['text']:
            related_content_found = True
            break
    assert related_content_found

    unrelated_results = synapse.recall_memory("今日の天気は？", limit=1)
    assert len(unrelated_results) > 0
    assert "天気" in unrelated_results[0]['text']

def test_recall_memory_with_context_filter(clean_db):
    synapse = clean_db
    synapse.add_experience("APIの設計を完了する。", "task")
    synapse.add_experience("昨日の会議で、新しい機能について話し合いました。", "chat")
    synapse.add_experience("テストコードを書く。", "task")
    synapse.sleep_process()

    results = synapse.recall_memory("今日やるべきこと", context={"node_type": "task"}, limit=2)
    assert len(results) > 0
    assert all(item['node_type'] == 'task' for item in results)
    assert any("APIの設計" in item['text'] for item in results)

def test_relationship_discovery_temporal(clean_db):
    synapse = clean_db
    exp1_id = synapse.add_experience("ユーザーがログインしました。", "log")
    time.sleep(0.01)
    exp2_id = synapse.add_experience("ログインに成功しました。", "log")
    
    synapse.sleep_process()

    cursor = synapse.connection.cursor()
    cursor.execute("SELECT id FROM memory_nodes WHERE text LIKE '%ログインに成功しました%' LIMIT 1")
    target_node = cursor.fetchone()
    
    assert target_node is not None
    related_mems = synapse.get_related_memories(target_node['id'], relationship_type='temporal_follows')
    
    assert len(related_mems) > 0
    assert any("ユーザーがログインしました" in synapse.get_node_by_id(rel['id'])['text'] for rel in related_mems)

def test_relationship_discovery_semantic(clean_db):
    synapse = clean_db
    synapse.add_experience("ChromaDBはベクトルデータベースです。", "statement")
    synapse.add_experience("埋め込み検索にChromaDBを使うのは良い選択です。", "statement")
    synapse.sleep_process()

    cursor = synapse.connection.cursor()
    cursor.execute("SELECT id FROM memory_nodes WHERE text LIKE '%ChromaDBはベクトルデータベースです%' LIMIT 1")
    source_node = cursor.fetchone()

    assert source_node is not None
    related_mems = synapse.get_related_memories(source_node['id'], relationship_type='semantic_similarity')
    
    assert len(related_mems) > 0
    assert any("埋め込み検索にChromaDBを使うのは良い選択です" in synapse.get_node_by_id(rel['id'])['text'] for rel in related_mems)

def test_get_project_context(clean_db):
    synapse = clean_db
    synapse.add_experience("APIの設計を完了する。", "task", {"project": "ProjA"})
    synapse.add_experience("フロントエンドを開発する。", "task", {"project": "ProjB"})
    synapse.add_experience("FastAPIのコードを書く。", "code", {"project": "ProjA"})
    synapse.sleep_process()

    context_a = synapse.get_project_context(project_name="ProjA")
    assert "APIの設計を完了する" in context_a['tasks']
    assert "FastAPIのコードを書く" in context_a['code_snippets']
    assert "フロントエンドを開発する" not in context_a['tasks']

    context_b = synapse.get_project_context(project_name="ProjB")
    assert "フロントエンドを開発する" in context_b['tasks']
    assert "APIの設計を完了する" not in context_b['tasks']