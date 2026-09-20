"""End-to-end test of the GUIDE framework — runs the full analyst workflow on Cora."""

import sys
import json

sys.path.insert(0, ".")

from arcade.data.loader import GraphStore
from arcade.engine.workspace import Workspace
from arcade.engine.scoring import ScoringEngine
from arcade.paradigms import ParadigmRegistry
from arcade.paradigms.detectors import run_detector
from arcade.knowledge import ArtifactRegistry
from arcade.models import AnalystState, EvidenceType
from arcade.tools import explore, measure, diagnose, investigate, compose


def main():
    print("=" * 60)
    print("GUIDE End-to-End Test — Cora Dataset")
    print("=" * 60)

    # Step 0: Load dataset
    print("\n[1/8] Loading Cora dataset...")
    gs = GraphStore()
    info = gs.load_dataset("cora")
    print(f"  Loaded: {info}")

    ws = Workspace(gs)

    # Step 1: Explore
    print("\n[2/8] Exploring graph...")
    ws.transition(AnalystState.EXPLORE)
    result = explore.graph_profile(ws)
    print(f"  {result.summary}")

    result = explore.degree_distribution(ws)
    print(f"  {result.summary}")

    result = explore.feature_stats(ws)
    print(f"  {result.summary}")

    # Step 2: Select paradigms
    print("\n[3/8] Selecting applicable paradigms...")
    pr = ParadigmRegistry()
    applicable = pr.applicable(ws.profile)
    print(f"  Applicable: {[p.name for p in applicable]}")

    # Step 3: Run detectors
    print("\n[4/8] Running paradigm detectors...")
    ws.transition(AnalystState.MEASURE)
    for p in applicable[:5]:  # Run 5 for speed
        ws.scoring.register_paradigm(p)
        try:
            scores = run_detector(p.detector_name, gs, epochs=30)
            ws.scoring.set_scores(p.name, scores)
            print(f"  ✓ {p.name}: mean={scores.mean():.4f}, p95={__import__('numpy').percentile(scores, 95):.4f}")
        except Exception as e:
            print(f"  ✗ {p.name}: {e}")

    # Step 4: Diagnose
    print("\n[5/8] Diagnosing score distributions...")
    ws.transition(AnalystState.DIAGNOSE)
    for name in ws.scoring.available_paradigms:
        result = diagnose.score_distribution(ws, name)
        print(f"  {name}: {result.data.get('num_modes', 0)} modes, "
              f"{'bimodal' if result.data.get('is_bimodal') else 'unimodal'}, "
              f"{'TAIL SPIKE' if result.data.get('tail_spike') else 'no spike'}")

        result = diagnose.degree_correlation(ws, name)
        print(f"    degree ρ={result.data['pearson_rho']:.4f} "
              f"{'← ARTIFACT' if result.data.get('is_degree_artifact') else '← OK'}")

    # Step 5: Compute weights
    print("\n[6/8] Computing GUIDE weights...")
    weights = ws.scoring.compute_weights(gs.profile(), gs.degrees(), gs.communities())
    for name, w in weights.items():
        print(f"  {name}: α={w.alpha}, β={w.beta:.3f}, γ={w.gamma:.3f}, w={w.w:.4f}")

    # Step 6: Find convergent nodes
    print("\n[7/8] Finding convergent nodes...")
    ws.transition(AnalystState.INVESTIGATE)
    result = investigate.find_convergent_nodes(ws, min_paradigms=2, top_k=10)
    print(f"  {result.summary}")

    if result.data.get("nodes"):
        top_nodes = [n["node_id"] for n in result.data["nodes"][:5]]
        result = investigate.cross_paradigm_view(ws, top_nodes)
        print(f"  {result.summary}")

    # Step 7: Fused scores
    print("\n[8/8] Computing fused scores...")
    top = ws.scoring.top_k(10)
    print(f"  Top 10 nodes (fused):")
    for t in top:
        print(f"    Node {t['node_id']}: score={t['fused_score']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    info = ws.session_info()
    print(f"Session: {info['state']}, {info['paradigms_scored']} paradigms, "
          f"{info['evidence_count']} evidence items, {info['elapsed_seconds']}s elapsed")

    # Check against ground truth if available
    labels = gs.labels
    if labels is not None:
        import numpy as np
        fused = ws.scoring.fuse()
        top_50 = np.argsort(fused)[-50:]
        anomalous = set(np.where(labels > 0)[0])
        if anomalous:
            tp = len(set(top_50) & anomalous)
            print(f"\n  Ground truth: {len(anomalous)} anomalous nodes")
            print(f"  Precision@50: {tp}/{len(top_50)} = {tp/len(top_50):.2%}")

    print("\n✓ GUIDE framework test complete.")


if __name__ == "__main__":
    main()
