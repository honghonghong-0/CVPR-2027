# Is It the Material or the Light?

Code for "Is It the Material or the Light? Diagnosing Material–Lighting Entanglement in Vision Foundation Models" (anonymous submission).

`scripts/` contains rendering (Blender 4.2, `render_pairs.py`), feature extraction (`extract_feats.py`, `extract_qwen_vis.py`),
the diagnosis (`analyze_e2.py`, `p0_review.py`, `subspace_overlap.py`, `e3_*.py`), lighting subspace removal
(`analyze_xfer*.py`, `controls_proj.py`, `boot_proj.py`), environment lighting / real photographs (`run_hdri.sh`, `ml_*.py`)
and the ZeST experiment (`zest_*.py`). Paths are relative to the project root. A cleaned release with data will follow.
