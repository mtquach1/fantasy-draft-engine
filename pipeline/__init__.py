"""Data pipeline: source adapters -> quality gate -> versioned snapshot store -> event log -> recompute.

Everything here is plain Python plus SQLite (standard library). It is designed to be replayable: given the same snapshots and the same
events in the same order, it rebuilds the same state, so it can be tested without any network access."""
