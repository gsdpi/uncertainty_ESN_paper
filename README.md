---

# Can Reservoirs Sense the Shift: Out-of-Distribution Detection in Echo State Networks

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
📄 *"Can Reservoirs Sense the Shift: Out-of-Distribution Detection in Echo State Networks"*.
It provides the necessary scripts and data to **reproduce the results** presented in the study.

## Acknowledgment

This work is part of Grant **PID2020-115401GB-I00**, funded by **MCIN/AEI/10.13039/501100011033**.
