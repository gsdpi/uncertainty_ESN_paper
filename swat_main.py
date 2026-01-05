##################################################################
# Entry point for SWaT anomaly detection
# Runs KNN and PCA baselines and saves metrics
##################################################################

from pprint import pprint

from swat_knn import run_swat_knn
from swat_pca import run_swat_pca


def run_all(csv_path: str = None, auto_download: bool = True):
    print("\nRunning KNN baseline...")
    knn_results = run_swat_knn(csv_path=csv_path, auto_download=auto_download)

    print("\nRunning PCA baseline...")
    pca_results = run_swat_pca(csv_path=csv_path, auto_download=auto_download)

    print("\nSummary (KNN vs PCA):")
    print("KNN:")
    pprint(knn_results)
    print("PCA:")
    pprint(pca_results)

    return {"knn": knn_results, "pca": pca_results}


if __name__ == "__main__":
    run_all()
    input("\nPress ENTER to exit...")
