# Awesome Ophthalmology
A curated list of awesome AI developments for ophthalmology




## Table of Contents

<!-- MarkdownTOC depth=4 -->


-  [Datasets](#datasets)
      -  [Diabetic Retinopathy Classification](#diabetic-retinopathy-classification)
      -  [Glaucoma Classification ](#glaucoma-classification)
      -  [Retinal Vessel Segmentation](#retinal-vessel-segmentation)
      -  [Retinal Image Registration](#retinal-image-registration)

<!-- /MarkdownTOC -->




<a name="datasets"></a>
## Datasets

<a name="diabetic-retinopathy-classification"></a>
### Diabetic Retinopathy Classification
#### [Kaggle EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection/) 
* **Summary:** Large set of high-resolution retina images taken under a variety of imaging conditions. A left and right field is provided for every subject. Images are labeled with a subject id as well as either left or right
* **Labels:** A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4
* **Scale:** 	88,702 retinal fundus images (35,126 labelled for training and 53,576 unlabelled for Kaggle challenge evaluation)
<br/><br/>


#### [IDRiD (Indian Diabetic Retinopathy Image Dataset)](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid#files) 
* **Summary:** This dataset provides information on the disease severity of diabetic retinopathy, and diabetic macular edema for each image. This makes it perfect for development and evaluation of image analysis algorithms for early detection of diabetic retinopathy
* **Labels:** Dataset consists of: 1) segmentation labels for microaneurysms, haemorrhages, hard exudates and soft exudates, 2) labels for diabetic retinopathy and diabetic macular edema severity grade, 3) location of optic disk and fovea
* **Scale:** 516 images labelled images for disease grading and localization, and 81 images for segmentation
<br/><br/>


#### [MESSIDOR](https://www.adcis.net/en/third-party/messidor/) 
* **Summary:** The Messidor database has been established to facilitate studies on computer-assisted diagnoses of diabetic retinopathy
* **Labels:** Two diagnoses have been provided by the medical experts for each image: 1) retinopathy grade, 2) risk of macular edema
* **Scale:** ~1200 labelled retinal fundus images
<br/><br/>





<a name="glaucoma-classification"></a>
### Glaucoma Classification 
#### [REFUGE Challenge (Retinal Fundus Glaucoma Challenge)](https://refuge.grand-challenge.org/REFUGE2Details/) 
* **Summary:** REFUGE Challenge consists of THREE Tasks: 1) Classification of clinical Glaucoma, 2) Segmentation of Optic Disc and Cup, 3) Localization of Fovea (macular center)
* **Labels:** A clinician has labelled the presence of glaucoma, the location of the fovea, and the segmentation mask for the optic disk and cup
* **Scale:** 	1200 labelled retinal fundus images
<br/><br/>


#### [G1020](https://paperswithcode.com/dataset/g1020) 
* **Summary:** A large publicly available retinal fundus image dataset for glaucoma classification called G1020. The dataset is curated by conforming to standard practices in routine ophthalmology and it is expected to serve as standard benchmark dataset for glaucoma detection
* **Labels:** This dataset provides ground truth annotations for glaucoma diagnosis, optic disc and optic cup segmentation, vertical cup-to-disc ratio, size of neuroretinal rim in inferior, superior, nasal and temporal quadrants, and bounding box location for optic disc
* **Scale:** 	1020 labelled high resolution colour fundus images
<br/><br/>







<a name="retinal-vessel-segmentation"></a>
### Retinal Vessel Segmentation
#### [ORVS (Online Retinal image for Vessel Segmentation)](https://github.com/AbdullahSarhan/ICPRVessels/tree/main/Vessels-Datasets/ORVS) 
* **Summary:** The ORVS dataset is a dataset for retinal vessel segmentation
* **Labels:** For the training images, a single manual segmentation of the vasculature is available
* **Scale:** 	49 labelled retinal fundus images 
<br/><br/>


#### [DRIVE (Digital Retinal Images for Vessel Extraction)](https://drive.grand-challenge.org/) 
* **Summary:** The DRIVE database has been established to enable comparative studies on segmentation of blood vessels in retinal images
* **Labels:** For the training images, a single manual segmentation of the vasculature is available
* **Scale:** 	40 retinal fundus images (20 labelled for training and 20 unlabelled for Grand Challenge evaluation)
<br/><br/>

#### [STARE (Structured Analysis of the Retina)](https://cecas.clemson.edu/~ahoover/stare/) 
* **Summary:** The STARE dataset is a dataset for retinal vessel segmentation
* **Labels:** For each image, two hand labelled vessel networks are provided
* **Scale:** 	20 labelled retinal fundus images
<br/><br/>


#### [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) 
* **Summary:** The CHASE_DB1 dataset is a dataset for retinal vessel segmentation
* **Labels:** For each image, two hand labelled vessel networks are provided
* **Scale:** 	28 labelled retinal fundus images
<br/><br/>


#### [HRF](https://www5.cs.fau.de/research/data/fundus-images/) 
* **Summary:** The HRF dataset is a dataset for retinal vessel segmentation which comprises 45 images and is organized as 15 subsets. Each subset contains one healthy fundus image, one image of patient with diabetic retinopathy and one glaucoma image. The image sizes are 3,304 x 2,336, with a training/testing image split of 22/23
* **Labels:** For each image, a labelled vessel network is provided
* **Scale:** 45 labelled retinal fundus images (15 healthy, 15 diabetic retinopathy, 15 glaucoma)
<br/><br/>


#### [RITE (Retinal Images vessel Tree Extraction)](https://medicine.uiowa.edu/eye/rite-dataset) 
* **Summary:** The RITE (Retinal Images vessel Tree Extraction) is a database that enables comparative studies on segmentation or classification of arteries and veins on retinal fundus images, which is established based on the public available DRIVE database (Digital Retinal Images for Vessel Extraction)
* **Labels:** For the training images, a single manual segmentation of the vasculature is available
* **Scale:** 40 labelled retinal fundus images
<br/><br/>


#### [DR HAGIS (Diabetic Retinopathy, Hypertension, Age-related macular degeneration and Glacuoma ImageS)](https://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/) 
* **Summary:** The DR HAGIS database has been created to aid the development of vessel extraction algorithms suitable for retinal screening programs. Researchers are encouraged to test their segmentation algorithms using this database
* **Labels:** For the images, a single annotated segmentation of the vasculature is available
* **Scale:** 40 labelled retinal fundus images
<br/><br/>


<a name="retinal-image-registration"></a>
### Retinal Image Registration
#### [FIRE (Fundus Image Registration Dataset)](https://projects.ics.forth.gr/cvrl/fire/) 
* **Summary:** FIRE is a dataset for retinal image registration, annotated with ground truth data
* **Labels:** For each pair of images, ground truth registration control points are provided
* **Scale:** 	The dataset consists of 134 labelled fundus image pairs
<br/><br/>



