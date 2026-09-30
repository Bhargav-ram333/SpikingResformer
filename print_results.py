import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")

def load_csv(filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    print("==========================================================================")
    print("                   SPIKINGRESFORMER CBM EVALUATION RESULTS                ")
    print("==========================================================================")

    # 1. Feature Representation Summary
    feat_stats = load_csv("feature_statistics.csv")
    print("\n--- 1. FEATURE REPRESENTATION STATISTICS ---")
    for r in feat_stats:
        dim = r['Dimension']
        mean = float(r['Mean'])
        std = float(r['Std'])
        zero_frac = float(r['Zero_Fraction'])
        print(f"Feature: {r['Feature_Representation']:<35} | Dim: {dim:<5} | Mean: {mean:+.4f} | Zero Frac: {zero_frac:.2%}")

    # 2. Concept Probe Metrics
    concept_data = load_csv("concept_metrics.csv")
    accs = [float(r['Accuracy']) for r in concept_data]
    aucs = [float(r['ROC_AUC']) for r in concept_data]
    f1s = [float(r['F1_Score']) for r in concept_data]

    print("\n--- 2. CONCEPT PROBE METRICS SUMMARY (CUB-200-2011 Dataset) ---")
    print(f"Total Attributes Evaluated : {len(concept_data)}")
    print(f"Mean Concept Accuracy      : {sum(accs)/len(accs)*100:.2f}%")
    print(f"Mean ROC-AUC Score         : {sum(aucs)/len(aucs):.4f}")
    print(f"Mean F1 Score              : {sum(f1s)/len(f1s):.4f}")
    print(f"Accuracy Range             : {min(accs)*100:.2f}% - {max(accs)*100:.2f}%")

    # Sample top/bottom attributes
    sorted_by_auc = sorted(concept_data, key=lambda x: float(x['ROC_AUC']), reverse=True)
    print("\nTop 3 Best Decodable Concept Attributes:")
    for r in sorted_by_auc[:3]:
        print(f"  * {r['Attribute']:<40} | ROC-AUC: {float(r['ROC_AUC']):.4f} | Acc: {float(r['Accuracy'])*100:.2f}%")
    print("\nBottom 3 Lowest Decodable Concept Attributes:")
    for r in sorted_by_auc[-3:]:
        print(f"  * {r['Attribute']:<40} | ROC-AUC: {float(r['ROC_AUC']):.4f} | Acc: {float(r['Accuracy'])*100:.2f}%")

    # 3. Regularization Sweep
    reg_data = load_csv("regularization_sweep_6144d.csv")
    print("\n--- 3. REGULARIZATION SWEEP (6,144-d Concat Feature) ---")
    for r in reg_data:
        c_val = float(r['C'])
        auc = float(r['Mean_ROC_AUC'])
        med = float(r['Median_ROC_AUC'])
        print(f"C = {c_val:<7} | Mean ROC-AUC: {auc:.5f} | Median ROC-AUC: {med:.5f} | Converged: {r['Converged_Probes']}")

    # 4. Statistical Significance Tests
    stats_data = load_csv("statistical_tests.csv")
    print("\n--- 4. PAIRWISE STATISTICAL SIGNIFICANCE TESTS ---")
    for r in stats_data:
        comp = r['Comparison']
        diff = float(r['Mean_Diff'])
        p_val = float(r['Paired_t_p_value'])
        sig = r['Statistically_Significant']
        print(f"Comparison : {comp}")
        print(f"  -> Mean AUC Diff: {diff:+.5f} | p-value: {p_val:.4e} | Statistically Significant: {sig}")

if __name__ == '__main__':
    main()
