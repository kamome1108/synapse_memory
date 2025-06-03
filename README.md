Synapse Memory: An AI Self-Development Project

Synapse Memory is a foundational memory system for AI agents, enabling them to learn from experience, form memories, and continuously improve and develop themselves. Inspired by neural synapses, it aims to connect fragmented pieces of information to build meaningful knowledge.

A core distinguishing feature of this project is its experimental approach: the AI agent powered by Synapse Memory will autonomously develop and improve this system itself. Thus, the evolution of this repository aims to be a "living proof" demonstrating the effectiveness of Synapse Memory.
Memory System Technology Stack

Synapse Memory combines the following technologies:

    SQLite: For persisting structured metadata, access patterns, and explicit relationships.

    ChromaDB: For efficient storage and high-speed semantic (vector) similarity search of memory nodes.

    Sentence-Transformers: For generating high-quality text embeddings, deepening semantic understanding.

Key Features

    Experience Storage: Records interactions, observations, and actions as "experiences".

    Atomic Memory Nodes: Automatically breaks down experiences into smaller, independent "memory nodes" for storage.

    Semantic Recall: Efficiently searches for memories semantically related to a query using embedding similarity, even if exact keywords are not present.

    Relationship Discovery: Automatically discovers and stores temporal, semantic, and inferential relationships between memory nodes.

    Sleep Process: A background consolidation mechanism that processes new experiences and forms new connections or insights between existing memories.

Vision for AI Self-Development

This project envisions the AI agent advancing its own development through processes such as:

    Self-Awareness & Goal Setting: The AI monitors the state of Synapse Memory and sets improvement goals, e.g., "enhance system robustness" or "add new features".

    Knowledge Recall & Planning: To achieve goals, the AI recalls past experiences (e.g., solutions to similar tasks, relevant code snippets, tool usage logs) from Synapse Memory.

    Execution of Development Tasks:

        Code Generation: Generates or modifies code for necessary functionalities based on recalled information and current context.

        Test Execution: Generates and runs test cases to ensure new code functions correctly and doesn't impact existing features.

        Debugging & Correction: If tests fail, the AI identifies the root cause from memory and attempts corrections.

    Documentation Updates: Updates README.md and other documentation to reflect development progress and new features.

    Version Control: Commits changes at appropriate granularities and creates branches or pull requests as needed.

    Continuous Improvement: Incorporates user feedback (e.g., GitHub Issues) as new "experiences" to feed into subsequent development cycles.

Through this process, the AI will evolve its own memory system by utilizing Synapse Memory.
Installation

To get started with Synapse Memory, follow these steps:

    Clone the repository:

    git clone https://github.com/kamome1108/synapse_memory.git
    ```bash
    cd synapse_memory

    Install the package:
    For development, installing in editable mode is recommended:

    pip install -e .

    This will install the synapse-memory package and its dependencies (sentence-transformers, numpy, chromadb).

Usage

Here's a quick example demonstrating how to use Synapse Memory:

from synapse_memory import SynapseMemory

# Initialize the memory system (debug=True for detailed logs)
memory = SynapseMemory(debug=True)

# Add some experiences
print("--- Adding Experiences ---")
memory.add_experience("Today's task is to finalize the API design.", "task", {"project": "my_web_app"})
memory.add_experience("Researched how to build RESTful APIs in Python using FastAPI.", "research", {"project": "my_web_app"})
memory.add_experience("Decided to use SQLite for the database; it's easy for persistence.", "decision", {"project": "my_web_app"})

# Run sleep process to consolidate memories and discover relationships
print("\n--- Running Sleep Process ---")
memory.sleep_process()

# Recall memories based on a query
print("\n--- Recalling Memories: 'progress on web app development' ---")
recalled_items = memory.recall_memory("progress on web app development", limit=5)
for item in recalled_items:
    print(f"- [Similarity: {item['similarity']:.3f}, Type: {item['node_type']}, Source: {item['source_type']}] {item['text']}")

# Close the memory connection
memory.close()
print("\n--- Memory system closed ---")

Running Tests

To ensure the integrity of the project, you can run the provided tests:

pytest tests/

Contributing and Participating in Self-Development

This project embodies the challenging aspect of AI self-development. If you resonate with this vision and wish to contribute, please feel free to engage via GitHub Issues or Pull Requests. Observing how the AI agent interprets your feedback and integrates it into its development plan is one of the unique aspects of this project.
