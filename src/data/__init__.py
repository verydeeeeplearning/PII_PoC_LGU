"""
src/data - Data Pipeline Modules

Stage S1: Normalize & Parse (Architecture.md §5)
    from src.data.s1_parser import (
        parse_context_field,
        infer_pii_type,
        make_pk_file,
        make_pk_event,
        compute_parse_success_rate,
        apply_schema_registry,
    )

Legacy modules (backward-compatible):
    loader.py, preprocessor.py, merger.py, validator.py
"""
