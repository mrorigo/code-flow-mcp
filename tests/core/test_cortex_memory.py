import time

from code_flow.core.cortex_memory import CortexMemoryStore


def test_decay_and_score_bounds(tmp_path):
    store = CortexMemoryStore(persist_directory=str(tmp_path))
    record = store.add_memory(
        content="Use snake_case",
        memory_type="TRIBAL",
        decay_half_life_days=0.00001,
        decay_floor=0.1,
    )

    # Force old reinforcement time
    record.last_reinforced = int(time.time()) - 10_000
    decay = store._compute_decay(record.last_reinforced, record.decay_half_life_days, record.decay_floor)
    score = store._compute_memory_score(record.base_confidence, record.reinforcement_count, decay)

    assert decay >= record.decay_floor
    assert score >= 0


def test_query_memory_returns_results(tmp_path):
    store = CortexMemoryStore(persist_directory=str(tmp_path))
    store.add_memory(content="Always use UTC timestamps", memory_type="FACT")
    results = store.query_memory(query="timestamps", n_results=3)
    assert len(results) >= 1
