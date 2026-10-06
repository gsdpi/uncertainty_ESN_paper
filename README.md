# Can Reservoirs Sense the Shift? Out-of-Distribution Detection in Echo State Networks

Model uncertainty, arising from regions of the input distribution insufficiently represented during training, is known as *epistemic uncertainty* and can lead to significant performance degradation during deployment. This issue becomes especially critical in dynamic environments, where data distributions evolve over time, a phenomenon known as *domain shift*. Echo State Networks, widely used as soft sensors and for real-time edge applications, must maintain robustness under such conditions to ensure reliability and safety. This paper addresses these challenges by proposing a novel similarity score that quantifies the match between the dynamic evolution of the reservoir states during training and inference. By identifying deviations in reservoir behavior, the method provides an implicit indicator of epistemic uncertainty and potential out-of-distribution inputs. This approach enhances model confidence, interpretability, and adaptability in non-stationary scenarios, requiring no additional training and operating independently of model accuracy, making it lightweight and easily deployable.

## Authors

- [José M. Enguita](mailto:jmenguita@uniovi.es)
- [Sara Roos-Hoefgeest](mailto:sroos@uniovi.es)
- [Diego García](mailto:garciaperdiego@uniovi.es)
- [Abel A. Cuadrado](mailto:aacuadrado@uniovi.es)
- [Ignacio Díaz](mailto:idiaz@uniovi.es)

## Affiliation

All authors are with the **Department of Electrical Engineering, University of Oviedo**, 33204 Gijón, Spain.

(C) [GSDPI research group](https://gsdpi.edv.uniovi.es/webpage/ "website")

Contact us by email at [gsdpi@uniovi.es](mailto:gsdpi@uniovi.es)

## Description

This repository contains the code used in the paper
📄 *"Can Reservoirs Sense the Shift? Out-of-Distribution Detection in Echo State Networks"*.
It provides the necessary scripts and data to **reproduce the results** presented in the study.

## Acknowledgment

This work is part of Grant **PID2020-115401GB-I00**, funded by **MCIN/AEI/10.13039/501100011033**.

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Key dependencies: `reservoirpy`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `requests`, `openpyxl`.

### Reproducibility check

The ICANN ESN comparison (`icann_esn.py`) initially produced different classification results on two PCs. The results matched again after aligning the package versions, so the following versions are pinned in `requirements.txt`:

- `reservoirpy==0.4.1`
- `scikit-learn==1.6.1`
- `numpy==2.0.2`
- `scipy==1.13.1`

For a comparable run, install dependencies from `requirements.txt`.

## Usage

The repository contains two groups of scripts with different purposes.

### 1. Main method evaluation (Cases 1 & 2)

These scripts reproduce the core results of the paper: they train an ESN, compute the proposed similarity-based uncertainty score, and evaluate its **stability and performance as a function of the latent dimensionality $r$**.

| Script                   | Dataset                                | Test case |
| ------------------------ | -------------------------------------- | --------- |
| `icann_process_v2.py`  | DATAICANN (electrical fault detection) | Case 1    |
| `imwsha_process_v2.py` | IM-WSHA (human activity recognition)   | Case 2    |

Both scripts **automatically download their respective datasets** on the first run (using `requests`). You can also download and extract them manually:

- **DATAICANN**: [http://hdl.handle.net/10651/53461](http://hdl.handle.net/10651/53461)
- **IM-WSHA**: [https://portals.au.edu.pk/imc/Pages/Datasets.aspx](https://portals.au.edu.pk/imc/Pages/Datasets.aspx)

Run them independently:

```bash
python icann_process_v2.py
python imwsha_process_v2.py
```

Output figures (ROC curves, AUC vs. $r$, threshold vs. $r$, etc.) are saved in the `figures/` directory.

---

### 2. Comparative evaluation (ESN vs. baselines)

These scripts evaluate the proposed ESN-based uncertainty method against three baseline approaches — **k-NN**, **MiniRocket**, and **PCA** — on all three datasets (DATAICANN, IM-WSHA, and a synthetic dataset). They can be run in **any order**.

| Dataset   | ESN               | k-NN              | MiniRocket               | PCA               |
| --------- | ----------------- | ----------------- | ------------------------ | ----------------- |
| DATAICANN | `icann_esn.py`  | `icann_knn.py`  | `icann_minirocket.py`  | `icann_pca.py`  |
| IM-WSHA   | `imwsha_esn.py` | `imwsha_knn.py` | `imwsha_minirocket.py` | `imwsha_pca.py` |
| Synthetic | `synth_esn.py`  | `synth_knn.py`  | `synth_minirocket.py`  | `synth_pca.py`  |

Example:

```bash
python icann_esn.py
python icann_knn.py
# ... and so on
```

Each script saves its results to an Excel file in the working directory (e.g., `results_icann_esn.xlsx`, `results_imwsha_knn.xlsx`, `results_synth_pca.xlsx`, …).

---

### 3. Non-windowed reservoir-state baseline (Mahalanobis)

These experiments compare the proposed **window + SVD + KDE** method with a baseline that scores **instantaneous reservoir states without sliding windows**. The current non-windowed detector is Mahalanobis distance, fitted on the training reservoir states and evaluated on the test states.

| Dataset   | Script               | Results file                    |
| --------- | -------------------- | ------------------------------- |
| DATAICANN | `icann_internal.py`  | `results_icann_internal.xlsx`   |
| IM-WSHA   | `imwsha_internal.py` | `results_imwsha_internal.xlsx`  |
| Synthetic | `synth_internal.py`  | `results_synth_internal.xlsx`   |

Run the desired experiments independently:

```bash
python icann_internal.py
python imwsha_internal.py
python synth_internal.py
```

The proposed-method results are produced by the corresponding `*_esn.py` scripts in Section 2.

---

### 4. Generate comparison figures

To generate figures comparing the proposed ESN method with **k-NN, MiniRocket, and PCA**, first produce their Excel results as described in Section 2, then run:

```bash
python generate_result_graphs.py
```

To compare the proposed method with the **non-windowed Mahalanobis baseline**, produce the matching `*_esn.py` and `*_internal.py` result files, then run:

```bash
python generate_internal_result_graphs.py
```

This generates `comparison_internal_icann.png`, `comparison_internal_imwsha.png`, and `comparison_internal_synth.png` in `figures/`. The figures compare ROC AUC, AUPRC, and F1-score; DATAICANN results are grouped by signal configuration, and IM-WSHA results are summarized across subjects.
