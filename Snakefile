rule all:
    input:
        # Processed datasets
        "data/processed/bioactivity_processed.csv",
        "data/processed/bioactivity_filtered.csv",
        "data/processed/descriptors.csv",

        # Tables
        "results/tables/activity_summary.csv",
        "results/tables/regression_metrics.csv",
        "results/tables/classification_metrics.csv",
        "results/tables/random_forest_feature_importance.csv",
        "results/tables/roc_auc_test_results.csv",
        "results/tables/random_forest_regression_tuning.csv",
        "results/tables/random_forest_classification_tuning.csv",

        # Figures
        "results/figures/ki_distribution_raw.png",
        "results/figures/pki_distribution.png",
        "results/figures/pca_descriptor_space.png",
        "results/figures/tsne_descriptor_space.png",
        "results/figures/umap_descriptor_space.png",
        "results/figures/random_forest_feature_importance.png",
        "results/figures/random_forest_predicted_vs_observed.png",
        "results/figures/roc_curves_classification.png"

rule preprocess:
    input:
        "data/raw/bioactivity_dataset.csv"
    output:
        "data/processed/bioactivity_processed.csv"
    shell:
        "python scripts/02_preprocess_data.py"

rule filter:
    input:
        "data/processed/bioactivity_processed.csv"
    output:
        "data/processed/bioactivity_filtered.csv"
    shell:
        "python scripts/06_filter_outliers.py"

rule descriptors:
    input:
        "data/processed/bioactivity_filtered.csv"
    output:
        "data/processed/descriptors.csv"
    shell:
        "python scripts/03_generate_descriptors.py"

rule activity_plots:
    input:
        "data/processed/bioactivity_filtered.csv"
    output:
        "results/figures/ki_distribution_raw.png",
        "results/figures/pki_distribution.png",
        "results/tables/activity_summary.csv"
    shell:
        "python scripts/04_explore_activity_distribution.py"

rule pca:
    input:
        "data/processed/descriptors.csv"
    output:
        "results/figures/pca_descriptor_space.png"
    shell:
        "python scripts/05_pca_visualization.py"

rule tsne:
    input:
        "data/processed/descriptors.csv"
    output:
        "results/figures/tsne_descriptor_space.png"
    shell:
        "python scripts/07_tsne_visualization.py"

rule umap:
    input:
        "data/processed/descriptors.csv"
    output:
        "results/figures/umap_descriptor_space.png"
    shell:
        "python scripts/08_umap_visualization.py"

rule regression:
    input:
        "data/processed/descriptors.csv"
    output:
        "results/tables/regression_metrics.csv"
    shell:
        "python scripts/09_regression_models.py"

rule classification:
    input:
        "data/processed/descriptors.csv"
    output:
        "results/tables/classification_metrics.csv"
    shell:
        "python scripts/10_classification_models.py"

rule feature_importance:
    input:
        "data/processed/descriptors.csv"
    output:
        "results/figures/random_forest_feature_importance.png",
        "results/tables/random_forest_feature_importance.csv"
    shell:
        "python scripts/11_feature_importance.py"

rule regression_predictions:
    input:
        "data/processed/descriptors.csv"
    output:
        "results/figures/random_forest_predicted_vs_observed.png",
        "results/tables/random_forest_test_predictions.csv"
    shell:
        "python scripts/12_regression_predictions.py"

rule roc_curves:
    input:
        "data/processed/descriptors.csv"
    output:
        "results/figures/roc_curves_classification.png",
        "results/tables/roc_auc_test_results.csv"
    shell:
        "python scripts/13_roc_curves.py"

rule rf_tuning:
    input:
        "data/processed/descriptors.csv"
    output:
        "results/tables/random_forest_regression_tuning.csv",
        "results/tables/random_forest_classification_tuning.csv"
    shell:
        "python scripts/14_random_forest_tuning.py"
