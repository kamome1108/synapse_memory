Synapse Memory: AIによる自律開発プロジェクト

Synapse Memory は、AIエージェントが経験に基づいて学習し、記憶を形成し、そして自己を継続的に改善・開発していくための基盤となる記憶システムです。脳のシナプスに触発され、断片化された情報を結びつけ、意味のある知識を構築することを目指しています。

このプロジェクトの最大の特徴は、Synapse Memory を搭載したAIエージェント自身が、このシステム自身の開発と改善を自律的に行うという実験的なアプローチを採用している点です。つまり、このリポジトリの進化自体が、Synapse Memory の有効性を示す「生きた実績」となることを目指しています。
記憶システムの技術スタック

Synapse Memory は、以下の技術を組み合わせています。

    SQLite: メモリのメタデータ、アクセスパターン、明示的な関係性などの構造化された情報を永続化します。

    ChromaDB: 記憶ノードの効率的なベクトル検索と高速なセマンティック（意味論的）類似性検索を実現します。

    Sentence-Transformers: 高品質なテキスト埋め込み（Embedding）を生成し、意味的な理解を深めます。

主な機能

    経験の保存 (Experience Storage): ユーザーとの対話、観測、行動などを「経験」として記録します。

    原子記憶ノード化 (Atomic Memory Nodes): 経験を、より小さく、独立した「記憶ノード」に自動的に分解・格納します。

    セマンティックな想起 (Semantic Recall): 埋め込みベクトルを用いた類似性検索により、クエリと意味的に関連性の高い記憶を効率的に検索します。

    関係性の発見 (Relationship Discovery): 記憶ノード間の時間的、意味的、推論的な関係性を自動的に発見し、保存します。

    スリーププロセス (Sleep Process): バックグラウンドで動作する統合メカニズム。新しい経験を処理し、既存の記憶間の新たな接続や洞察を形成します。

AIによる自律開発のビジョン

このプロジェクトは、AIエージェントが以下のようなプロセスで自己開発を進めることを想定しています。

    自己認識と目標設定: AIはSynapse Memoryの状態を監視し、例えば「システムの堅牢性を高める」「新しい機能を追加する」といった改善目標を自ら設定します。

    知識の想起と計画: 目標達成のために、過去の経験（類似するタスクの解決策、関連するコードスニペット、ツール利用の記録など）をSynapse Memoryから想起します。

    開発タスクの実行:

        コード生成: 記憶から得た情報と、現在の文脈に基づいて、必要な機能のコードを生成または修正します。

        テストの実施: 生成したコードが正しく動作するか、既存の機能に影響を与えないかを確認するためのテストコードを生成し、実行します。

        デバッグと修正: テストが失敗した場合、その原因を記憶から特定し、修正を試みます。

    ドキュメントの更新: 開発の進捗や新しい機能に合わせて、README.md やその他のドキュメントを更新します。

    バージョン管理: 適切な粒度で変更をコミットし、必要に応じてブランチやプルリクエストを作成します。

    継続的改善: ユーザーからのフィードバック（GitHub Issuesなど）も「経験」として取り込み、次なる改善サイクルに繋げます。

このプロセスを通じて、AIはSynapse Memoryを使いながら、自身のメモリシステムそのものを進化させていきます。
インストール

Synapse Memory のコードを取得し、セットアップする方法は以下の通りです。

    リポジトリをクローン:

    git clone [https://github.com/kamome1108/synapse_memory.git](https://github.com/kamome1108/synapse_memory.git)
    ```bash
    cd synapse_memory

    パッケージをインストール:
    開発環境で利用する場合、編集可能モードでのインストールを推奨します。

    pip install -e .

    これにより、synapse-memory パッケージと、その依存関係（sentence-transformers, numpy, chromadb）がインストールされます。

使用方法

基本的な Synapse Memory の利用例です。このコードを実行すると、synapse_memory.db と synapse_chroma_db が生成されます。

from synapse_memory import SynapseMemory

# メモリシステムを初期化 (debug=Trueで詳細なログを出力)
memory = SynapseMemory(debug=True)

# いくつかの経験を追加
print("--- 経験の追加 ---")
memory.add_experience("今日のタスクは、APIの設計を完了することです。", "task", {"project": "my_web_app"})
memory.add_experience("FastAPIを使ってPythonでRESTful APIを構築する方法を調査しました。", "research", {"project": "my_web_app"})
memory.add_experience("データベースはSQLiteを使うことに決めました。永続化が簡単です。", "decision", {"project": "my_web_app"})

# スリーププロセスを実行し、記憶を統合し、関係性を発見
print("\n--- スリーププロセスの実行 ---")
memory.sleep_process()

# クエリに基づいて記憶を想起
print("\n--- 記憶の想起: 'ウェブアプリ開発の進捗について' ---")
recalled_items = memory.recall_memory("ウェブアプリ開発の進捗について", limit=5)
for item in recalled_items:
    print(f"- [類似度: {item['similarity']:.3f}, タイプ: {item['node_type']}, ソース: {item['source_type']}] {item['text']}")

# メモリ接続を閉じる
memory.close()
print("\n--- メモリシステムを閉じました ---")

テストの実行

プロジェクトの健全性を確認するためには、以下のコマンドでテストを実行できます。

pytest tests/

貢献と自律開発への参加

このプロジェクトは、AIによる自律開発という挑戦的な側面を持っています。もしあなたがこのビジョンに共感し、貢献したい場合は、ぜひGitHubのIssueやPull Requestを通じて参加してください。AIエージェントがあなたのフィードバックをどのように解釈し、自身の開発計画に組み込むかを見るのも、このプロジェクトの醍醐味の一つです。
ライセンス

このプロジェクトは、MITライセンスの下で公開されています。詳細については、LICENSE ファイルをご覧ください。