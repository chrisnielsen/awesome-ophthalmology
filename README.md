# Awesome Ophthalmology
A curated list of awesome AI developments for ophthalmology




## Table of Contents

<!-- MarkdownTOC depth=4 -->


-  [Datasets](#datasets)
      -  [Diabetic Retinopathy Classification](#datasets-diabetic-retinopathy-classification)
      -  [Glaucoma Classification ](#datasets-glaucoma-classification)
      -  [Retinal Image Registration](#datasets-retinal-image-registration)
      -  [Retinal Vessel Segmentation](#datasets-retinal-vessel-segmentation)
-  [Papers](#papers)
      -  [Retinal Vessel Segmentation](#papers-retinal-vessel-segmentation)

<!-- /MarkdownTOC -->




<a name="datasets"></a>
## Datasets

--- 
<a name="datasets-diabetic-retinopathy-classification"></a>
### Diabetic Retinopathy Classification
---
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




--- 
<a name="datasets-glaucoma-classification"></a>
### Glaucoma Classification
--- 
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


--- 
<a name="datasets-retinal-image-registration"></a>
### Retinal Image Registration
---
#### [FIRE (Fundus Image Registration Dataset)](https://projects.ics.forth.gr/cvrl/fire/) 
* **Summary:** FIRE is a dataset for retinal image registration, annotated with ground truth data
* **Labels:** For each pair of images, ground truth registration control points are provided
* **Scale:** 	The dataset consists of 134 labelled fundus image pairs
<br/><br/>



--- 
<a name="datasets-retinal-vessel-segmentation"></a>
### Retinal Vessel Segmentation
---
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




<a name="papers"></a>
## Papers


---
<a name="generative-models"></a>
### Generative Models
---
* 2021 - Explaining in Style: Training a GAN to explain a classifier in StyleSpace (Lang et al.) [(paper)](https://arxiv.org/pdf/2104.13369v1.pdf) 
* 2021 - Jekyll: Attacking Medical Image Diagnostics using Deep Generative Models (Mangaokar et al.) [(paper)](https://arxiv.org/pdf/2104.02107v1.pdf) 
* 2021 - VTGAN: Semi-supervised Retinal Image Synthesis and Disease Prediction using Vision Transformers (Kamran et al.) [(paper)](https://arxiv.org/pdf/2104.06757v1.pdf) [(code)](https://github.com/SharifAmit/VTGAN)
* 2020 - Analysis of Macula on Color Fundus Images Using Heightmap Reconstruction Through Deep Learning (Tahghighi et al.) [(paper)](https://arxiv.org/pdf/2012.14140v1.pdf) 
* 2020 - Attention2AngioGAN: Synthesizing Fluorescein Angiography from Retinal Fundus Images using Generative Adversarial Networks (Kamran et al.) [(paper)](https://arxiv.org/pdf/2007.09191v1.pdf) [(code)](https://github.com/SharifAmit/Attention2Angio)
* 2020 - Cross-Spectral Periocular Recognition with Conditional Adversarial Networks (Hernandez-Diaz et al.) [(paper)](https://arxiv.org/pdf/2008.11604v1.pdf) 
* 2020 - Dual In-painting Model for Unsupervised Gaze Correction and Animation in the Wild (Zhang et al.) [(paper)](https://arxiv.org/pdf/2008.03834v1.pdf) [(code)](https://github.com/zhangqianhui/GazeAnimation)
* 2020 - Forecasting Irreversible Disease via Progression Learning (Wu et al.) [(paper)](https://arxiv.org/pdf/2012.11107v2.pdf) 
* 2020 - Fundus2Angio: A Conditional GAN Architecture for Generating Fluorescein Angiography Images from Retinal Fundus Photography (Kamran et al.) [(paper)](https://arxiv.org/pdf/2005.05267v2.pdf) [(code)](https://github.com/SharifAmit/Fundus2Angio)
* 2020 - Generating Fundus Fluorescence Angiography Images from Structure Fundus Images Using Generative Adversarial Networks (Li et al.) [(paper)](https://arxiv.org/pdf/2006.10216v1.pdf) 
* 2020 - Heightmap Reconstruction of Macula on Color Fundus Images Using Conditional Generative Adversarial Networks (Tahghighi et al.) [(paper)](https://arxiv.org/pdf/2009.01601v4.pdf) [(code)](https://github.com/PeymanTahghighi/FundusDeepLearning)
* 2020 - Learning Two-Stream CNN for Multi-Modal Age-related Macular Degeneration Categorization (Wang et al.) [(paper)](https://arxiv.org/pdf/2012.01879v1.pdf) 
* 2020 - Medical Image Generation using Generative Adversarial Networks (Singh et al.) [(paper)](https://arxiv.org/pdf/2005.10687v1.pdf) 
* 2020 - Resolution enhancement and realistic speckle recovery with generative adversarial modeling of micro-optical coherence tomography (Liang et al.) [(paper)](http://arxiv.org/pdf/2003.06035v2.pdf) 
* 2020 - Towards the Next Generation of Retinal Neuroprosthesis: Visual Computation with Spikes (Yu et al.) [(paper)](http://arxiv.org/pdf/2001.04064v1.pdf) 
* 2019 - Annotation-Free Cardiac Vessel Segmentation via Knowledge Transfer from Retinal Images (Yu et al.) [(paper)](https://arxiv.org/pdf/1907.11483v1.pdf) 
* 2019 - Blind Inpainting of Large-scale Masks of Thin Structures with Adversarial and Reinforcement Learning (Chen et al.) [(paper)](http://arxiv.org/pdf/1912.02470v1.pdf) 
* 2019 - Digital resolution enhancement in low transverse sampling optical coherence tomography angiography using deep learning (Zhou et al.) [(paper)](http://arxiv.org/pdf/1910.01344v2.pdf) 
* 2019 - Eliminating Shadow Artifacts via Generative Inpainting Networks to Quantify Vascular Changes of the Choroid (Zhang et al.) [(paper)](http://arxiv.org/pdf/1907.01271v2.pdf) 
* 2019 - Noise as Domain Shift: Denoising Medical Images by Unpaired Image Translation (Manakov et al.) [(paper)](https://arxiv.org/pdf/1910.02702v1.pdf) [(code)](https://github.com/IljaManakov/HDcycleGAN)
* 2019 - SkrGAN: Sketching-rendering Unconditional Generative Adversarial Networks for Medical Image Synthesis (Zhang et al.) [(paper)](https://arxiv.org/pdf/1908.04346v1.pdf) 
* 2019 - Synthesizing New Retinal Symptom Images by Multiple Generative Models (Liu et al.) [(paper)](http://arxiv.org/pdf/1902.04147v1.pdf) [(code)](https://github.com/huckiyang/EyeNet-GANs)
* 2018 - Generative Adversarial Network for Medical Images (MI-GAN) (Iqbal et al.) [(paper)](http://arxiv.org/pdf/1810.00551v1.pdf) [(code)](https://github.com/hazratali/MI-GAN)
* 2018 - Generative Adversarial Network in Medical Imaging: A Review (Yi et al.) [(paper)](https://arxiv.org/pdf/1809.07294v4.pdf) [(code)](https://github.com/xinario/awesome-gan-for-medical-imaging)
* 2018 - Semi-Supervised Deep Learning for Abnormality Classification in Retinal Images (Lecouat et al.) [(paper)](http://arxiv.org/pdf/1812.07832v1.pdf) [(code)](https://github.com/theidentity/Improved-GAN-PyTorch)
* 2017 - LOGAN: Membership Inference Attacks Against Generative Models (Hayes et al.) [(paper)](http://arxiv.org/pdf/1705.07663v4.pdf) [(code)](https://github.com/jhayes14/gen_mem_inf)
* 2017 - Synthetic Medical Images from Dual Generative Adversarial Networks (Guibas et al.) [(paper)](http://arxiv.org/pdf/1709.01872v3.pdf) [(code)](https://github.com/HarshaVardhanVanama/Synthetic-Medical-Images)
<br/><br/>

---
<a name="glaucoma"></a>
### Glaucoma
---
* 2021 - Dynamic region proposal networks for semantic segmentation in automated glaucoma screening (Shah et al.) [(paper)](https://arxiv.org/pdf/2105.11364v1.pdf) 
* 2021 - Rapid Classification of Glaucomatous Fundus Images (Singh et al.) [(paper)](https://arxiv.org/pdf/2102.04400v1.pdf) 
* 2021 - Sample Efficient Learning of Image-Based Diagnostic Classifiers Using Probabilistic Labels (Vega et al.) [(paper)](https://arxiv.org/pdf/2102.06164v1.pdf) 
* 2020 - A Macro-Micro Weakly-supervised Framework for AS-OCT Tissue Segmentation (Ning et al.) [(paper)](https://arxiv.org/pdf/2007.10007v1.pdf) 
* 2020 - CNN-based approach for glaucoma diagnosis using transfer learning and LBP-based data augmentation (Maheshwari et al.) [(paper)](http://arxiv.org/pdf/2002.08013v1.pdf) 
* 2020 - Difficulty-aware Glaucoma Classification with Multi-Rater Consensus Modeling (Yu et al.) [(paper)](https://arxiv.org/pdf/2007.14848v1.pdf) 
* 2020 - EGDCL: An Adaptive Curriculum Learning Framework for Unbiased Glaucoma Diagnosis (Zhao et al.) [(paper)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660188.pdf) 
* 2020 - Large-scale machine learning-based phenotyping significantly improves genomic discovery for optic nerve head morphology (Alipanahi et al.) [(paper)](https://arxiv.org/pdf/2011.13012v1.pdf) 
* 2020 - Leveraging Undiagnosed Data for Glaucoma Classification with Teacher-Student Learning (Wu et al.) [(paper)](https://arxiv.org/pdf/2007.11355v1.pdf) 
* 2020 - One-Vote Veto: Semi-Supervised Learning for Low-Shot Glaucoma Diagnosis (Fan et al.) [(paper)](https://arxiv.org/pdf/2012.04841v3.pdf) [(code)](https://github.com/ruirangerfan/ovv_self_training)
* 2020 - Open-Narrow-Synechiae Anterior Chamber Angle Classification in AS-OCT Sequences (Hao et al.) [(paper)](https://arxiv.org/pdf/2006.05367v1.pdf) 
* 2020 - RetiNerveNet: Using Recursive Deep Learning to Estimate Pointwise 24-2 Visual Field Data based on Retinal Structure (Datta et al.) [(paper)](https://arxiv.org/pdf/2010.07488v1.pdf) 
* 2019 - AxoNet: an AI-based tool to count retinal ganglion cell axons (Ritch et al.) [(paper)](http://arxiv.org/pdf/1908.02919v1.pdf) 
* 2019 - Evaluation of an AI system for the automated detection of glaucoma from stereoscopic optic disc photographs: the European Optic Disc Assessment Study (Rogers et al.) [(paper)](https://arxiv.org/pdf/1906.01272v1.pdf) 
* 2019 - Identification of primary angle-closure on AS-OCT images with Convolutional Neural Networks (Yuan et al.) [(paper)](https://arxiv.org/pdf/1910.10414v1.pdf) 
* 2019 - Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation (Ozbulak et al.) [(paper)](https://arxiv.org/pdf/1907.13124v1.pdf) [(code)](https://github.com/utkuozbulak/adaptive-segmentation-mask-attack)
* 2019 - TRk-CNN: Transferable Ranking-CNN for image classification of glaucoma, glaucoma suspect, and normal eyes (Jun et al.) [(paper)](https://arxiv.org/pdf/1905.06509v1.pdf) 
* 2019 - Task Decomposition and Synchronization for Semantic Biomedical Image Segmentation (Ren et al.) [(paper)](https://arxiv.org/pdf/1905.08720v2.pdf) 
* 2019 - Visual Field Prediction using Recurrent Neural Network (Park et al.) [(paper)](https://www.nature.com/articles/s41598-019-44852-6.pdf) 
* 2018 - 2sRanking-CNN: A 2-stage ranking-CNN for diagnosis of glaucoma from fundus images using CAM-extracted ROI as an intermediate input (Jun et al.) [(paper)](http://arxiv.org/pdf/1805.05727v2.pdf) 
* 2018 - A Deep Learning based Joint Segmentation and Classification Framework for Glaucoma Assesment in Retinal Color Fundus Images (Arunava et al.) [(paper)](http://arxiv.org/pdf/1808.01355v1.pdf) 
* 2018 - A Deep Learning based Joint Segmentation and Classification Framework for Glaucoma Assesment in Retinal Color Fundus Images (Chakravarty et al.) [(paper)](http://arxiv.org/pdf/1808.01355v1.pdf) 
* 2018 - A spatially varying change points model for monitoring glaucoma progression using visual field data (Berchuck et al.) [(paper)](http://arxiv.org/pdf/1811.11038v1.pdf) 
* 2018 - Deep Learning and Glaucoma Specialists: The Relative Importance of Optic Disc Features to Predict Glaucoma Referral in Fundus Photos (Phene et al.) [(paper)](https://arxiv.org/pdf/1812.08911v2.pdf) 
* 2018 - Diagnosing Glaucoma Progression with Visual Field Data Using a Spatiotemporal Boundary Detection Method (Berchuck et al.) [(paper)](http://arxiv.org/pdf/1805.11636v1.pdf) 
* 2018 - Disc-aware Ensemble Network for Glaucoma Screening from Fundus Image (Fu et al.) [(paper)](http://arxiv.org/pdf/1805.07549v1.pdf) [(code)](https://github.com/HzFu/MNet_DeepCDR)
* 2018 - Enhanced Optic Disk and Cup Segmentation with Glaucoma Screening from Fundus Images using Position encoded CNNs (Agrawal et al.) [(paper)](http://arxiv.org/pdf/1809.05216v1.pdf) [(code)](https://github.com/koriavinash1/Optic-Disk-Cup-Segmentation)
* 2018 - Forecasting Future Humphrey Visual Fields Using Deep Learning (Wen et al.) [(paper)](http://arxiv.org/pdf/1804.04543v1.pdf) [(code)](https://github.com/uw-biomedical-ml/hvfProgression)
* 2018 - Performance assessment of the deep learning technologies in grading glaucoma severity (Zhen et al.) [(paper)](http://arxiv.org/pdf/1810.13376v1.pdf) 
* 2018 - Stack-U-Net: Refinement Network for Image Segmentation on the Example of Optic Disc and Cup (Sevastopolsky et al.) [(paper)](http://arxiv.org/pdf/1804.11294v2.pdf) 
* 2018 - Web Applicable Computer-aided Diagnosis of Glaucoma Using Deep Learning (Kim et al.) [(paper)](http://arxiv.org/pdf/1812.02405v2.pdf) 
<br/><br/>

---
<a name="iris-segmentation"></a>
### Iris Segmentation
---
* 2021 - Semi-Supervised Learning for Eye Image Segmentation (Chaudhary et al.) [(paper)](https://arxiv.org/pdf/2103.09369v1.pdf) 
* 2021 - TEyeD: Over 20 million real-world eye images with Pupil, Eyelid, and Iris 2D and 3D Segmentations, 2D and 3D Landmarks, 3D Eyeball, Gaze Vector, and Eye Movement Types (Fuhl et al.) [(paper)](https://arxiv.org/pdf/2102.02115v1.pdf) 
* 2021 - Two-stage CNN-based wood log recognition (Wimmer et al.) [(paper)](https://arxiv.org/pdf/2101.04450v1.pdf) 
* 2020 - $pi_t$- Enhancing the Precision of Eye Tracking using Iris Feature Motion Vectors (Chaudhary et al.) [(paper)](https://arxiv.org/pdf/2009.09348v1.pdf) 
* 2020 - An approach to human iris recognition using quantitative analysis of image features and machine learning (Khuzani et al.) [(paper)](https://arxiv.org/pdf/2009.05880v1.pdf) 
* 2020 - Complex-valued Iris Recognition Network (Nguyen et al.) [(paper)](https://arxiv.org/pdf/2011.11198v1.pdf) 
* 2020 - Cycle-consistent Generative Adversarial Networks for Neural Style Transfer using data from Chang'E-4 (Curtó et al.) [(paper)](https://arxiv.org/pdf/2011.11627v1.pdf) [(code)](https://github.com/decurtoidiaz/ce4)
* 2020 - Data Segmentation via t-SNE, DBSCAN, and Random Forest (DeLise) [(paper)](https://arxiv.org/pdf/2010.13682v2.pdf) [(code)](https://github.com/jefferythewind/tsne_dbscan_rf)
* 2020 - Deep Learning-based Pupil Center Detection for Fast and Accurate Eye Tracking System (Lee et al.) [(paper)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640035.pdf) 
* 2020 - EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking (Kothari et al.) [(paper)](https://arxiv.org/pdf/2007.09600v1.pdf) [(code)](https://bitbucket.org/RSKothari/ellseg)
* 2020 - Iris segmentation techniques to recognize the behavior of a vigilant driver (Baba) [(paper)](https://arxiv.org/pdf/2005.02450v1.pdf) 
* 2020 - Micro Stripes Analyses for Iris Presentation Attack Detection (Fang et al.) [(paper)](https://arxiv.org/pdf/2010.14850v2.pdf) 
* 2020 - Morton Filters for Superior Template Protection for Iris Recognition (Raja et al.) [(paper)](https://arxiv.org/pdf/2001.05290v1.pdf) 
* 2020 - Multispectral Biometrics System Framework: Application to Presentation Attack Detection (Spinoulas et al.) [(paper)](https://arxiv.org/pdf/2006.07489v1.pdf) 
* 2020 - On Benchmarking Iris Recognition within a Head-mounted Display for AR/VR Application (Boutros et al.) [(paper)](https://arxiv.org/pdf/2010.11700v1.pdf) 
* 2020 - Open Source Iris Recognition Hardware and Software with Presentation Attack Detection (Fang et al.) [(paper)](https://arxiv.org/pdf/2008.08220v1.pdf) [(code)](https://github.com/CVRL/RaspberryPiOpenSourceIris)
* 2020 - Privacy-Preserving Eye Videos using Rubber Sheet Model (Chaudhary et al.) [(paper)](https://arxiv.org/pdf/2004.01792v1.pdf) 
* 2020 - RIT-Eyes: Rendering of near-eye images for eye-tracking applications (Nair et al.) [(paper)](https://arxiv.org/pdf/2006.03642v1.pdf) 
* 2020 - Recognition Oriented Iris Image Quality Assessment in the Feature Space (Wang et al.) [(paper)](https://arxiv.org/pdf/2009.00294v2.pdf) [(code)](https://github.com/Debatrix/DFSNet)
* 2020 - Reconstruction and Quantification of 3D Iris Surface for Angle-Closure Glaucoma Detection in Anterior Segment OCT (Hao et al.) [(paper)](https://arxiv.org/pdf/2006.05179v1.pdf) [(code)](https://github.com/iMED-Lab/WRB-Net)
* 2020 - Resist : Reconstruction of irises from templates (Ahmad et al.) [(paper)](https://arxiv.org/pdf/2007.15850v2.pdf) [(code)](https://github.com/sohaib50k/RESIST-Iris-template-reconstruction)
* 2020 - SIP-SegNet: A Deep Convolutional Encoder-Decoder Network for Joint Semantic Segmentation and Extraction of Sclera, Iris and Pupil based on Periocular Region Suppression (Hassan et al.) [(paper)](https://arxiv.org/pdf/2003.00825v1.pdf) 
* 2020 - UFPR-Periocular: A Periocular Dataset Collected by Mobile Devices in Unconstrained Scenarios (Zanlorensi et al.) [(paper)](https://arxiv.org/pdf/2011.12427v1.pdf) 
* 2019 - A Resource-Efficient Embedded Iris Recognition System Using Fully Convolutional Networks (Tann et al.) [(paper)](https://arxiv.org/pdf/1909.03385v1.pdf) [(code)](https://github.com/scale-lab/FCNiris)
* 2019 - Biometrics Recognition Using Deep Learning: A Survey (Minaee et al.) [(paper)](https://arxiv.org/pdf/1912.00271v3.pdf) 
* 2019 - Deep Learning Algorithms to Isolate and Quantify the Structures of the Anterior Segment in Optical Coherence Tomography Images (Pham et al.) [(paper)](https://arxiv.org/pdf/1909.00331v1.pdf) 
* 2019 - Deep Neural Network and Data Augmentation Methodology for off-axis iris segmentation in wearable headsets (Varkarakis et al.) [(paper)](http://arxiv.org/pdf/1903.00389v1.pdf) 
* 2019 - Deep Representations for Cross-spectral Ocular Biometrics (Zanlorensi et al.) [(paper)](https://arxiv.org/pdf/1911.09509v1.pdf) 
* 2019 - DeepIrisNet2: Learning Deep-IrisCodes from Scratch for Segmentation-Robust Visible Wavelength and Near Infrared Iris Recognition (Gangwar et al.) [(paper)](http://arxiv.org/pdf/1902.05390v1.pdf) 
* 2019 - Eyenet: Attention based Convolutional Encoder-Decoder Network for Eye Region Segmentation (Kansal et al.) [(paper)](https://arxiv.org/pdf/1910.03274v1.pdf) 
* 2019 - Influence of segmentation on deep iris recognition performance (Lozej et al.) [(paper)](https://arxiv.org/pdf/1901.10431v2.pdf) 
* 2019 - Iris R-CNN: Accurate Iris Segmentation in Non-cooperative Environment (Feng et al.) [(paper)](http://arxiv.org/pdf/1903.10140v1.pdf) 
* 2019 - Iris Recognition for Personal Identification using LAMSTAR neural network (Homayon et al.) [(paper)](https://arxiv.org/pdf/1907.12145v1.pdf) 
* 2019 - Iris Recognition with Image Segmentation Employing Retrained Off-the-Shelf Deep Neural Networks (Kerrigan et al.) [(paper)](http://arxiv.org/pdf/1901.01028v1.pdf) [(code)](https://github.com/CVRL/iris-recognition-OTS-DNN)
* 2019 - Iris Verification with Convolutional Neural Network and Unit-Circle Layer (Spetlik et al.) [(paper)](https://arxiv.org/pdf/1906.09472v2.pdf) 
* 2019 - Joint Iris Segmentation and Localization Using Deep Multi-task Learning Framework (Wang et al.) [(paper)](https://arxiv.org/pdf/1901.11195v2.pdf) [(code)](https://github.com/xiamenwcy/IrisParseNet)
* 2019 - Learning scale-variant features for robust iris authentication with deep learning based ensemble framework (Zheng et al.) [(paper)](https://arxiv.org/pdf/1912.00756v2.pdf) 
* 2019 - Learning-Free Iris Segmentation Revisited: A First Step Toward Fast Volumetric Operation Over Video Samples (Kinnison et al.) [(paper)](http://arxiv.org/pdf/1901.01575v1.pdf) [(code)](https://github.com/jeffkinnison/florin-iris)
* 2019 - Ocular Recognition Databases and Competitions: A Survey (Zanlorensi et al.) [(paper)](https://arxiv.org/pdf/1911.09646v1.pdf) 
* 2019 - OpenEDS: Open Eye Dataset (Garbin et al.) [(paper)](https://arxiv.org/pdf/1905.03702v2.pdf) 
* 2019 - Post-Mortem Iris Recognition Resistant to Biological Eye Decay Processes (Trokielewicz et al.) [(paper)](https://arxiv.org/pdf/1912.02512v1.pdf) 
* 2019 - Post-mortem Iris Recognition with Deep-Learning-based Image Segmentation (Trokielewicz et al.) [(paper)](https://arxiv.org/pdf/1901.01708v2.pdf) [(code)](https://github.com/aczajka/iris-recognition---pm-diseased-human-driven-bsif)
* 2019 - Realtime and Accurate 3D Eye Gaze Capture with DCNN-based Iris and Pupil Segmentation (Wang et al.) [(paper)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8818661) [(code)](https://github.com/1996scarlet/Laser-Eye)
* 2019 - Relevant features for Gender Classification in NIR Periocular Images (Viedma et al.) [(paper)](http://arxiv.org/pdf/1904.12007v1.pdf) 
* 2019 - Segmentation-Aware and Adaptive Iris Recognition (Wang et al.) [(paper)](https://arxiv.org/pdf/2001.00989v1.pdf) 
* 2019 - Simultaneous Iris and Periocular Region Detection Using Coarse Annotations (Lucio et al.) [(paper)](https://arxiv.org/pdf/1908.00069v1.pdf) 
* 2019 - ThirdEye: Triplet Based Iris Recognition without Normalization (Ahmad et al.) [(paper)](https://arxiv.org/pdf/1907.06147v1.pdf) [(code)](https://github.com/sohaib50k/ThirdEye---Iris-recognition-using-triplets)
* 2018 - A Survey on Periocular Biometrics Research (Alonso-Fernandez et al.) [(paper)](http://arxiv.org/pdf/1810.03360v1.pdf) 
* 2018 - Assessment of iris recognition reliability for eyes affected by ocular pathologies (Trokielewicz et al.) [(paper)](http://arxiv.org/pdf/1809.00206v1.pdf) 
* 2018 - Cataract influence on iris recognition performance (Trokielewicz et al.) [(paper)](http://arxiv.org/pdf/1809.00211v1.pdf) 
* 2018 - Data-Driven Segmentation of Post-mortem Iris Images (Trokielewicz et al.) [(paper)](http://arxiv.org/pdf/1807.04154v1.pdf) 
* 2018 - Database of iris images acquired in the presence of ocular pathologies and assessment of iris recognition reliability for disease-affected eyes (Trokielewicz et al.) [(paper)](http://arxiv.org/pdf/1809.00212v1.pdf) 
* 2018 - Fully Convolutional Networks and Generative Adversarial Networks Applied to Sclera Segmentation (Lucio et al.) [(paper)](http://arxiv.org/pdf/1806.08722v3.pdf) 
* 2018 - Implications of Ocular Pathologies for Iris Recognition Reliability (Trokielewicz et al.) [(paper)](http://arxiv.org/pdf/1809.00168v1.pdf) 
* 2018 - Iris Recognition After Death (Trokielewicz et al.) [(paper)](http://arxiv.org/pdf/1804.01962v2.pdf) 
* 2018 - Iris Recognition with a Database of Iris Images Obtained in Visible Light Using Smartphone Camera (Trokielewicz) [(paper)](http://arxiv.org/pdf/1809.00214v1.pdf) 
* 2018 - Iris and periocular recognition in arabian race horses using deep convolutional neural networks (Trokielewicz et al.) [(paper)](http://arxiv.org/pdf/1809.00213v1.pdf) 
* 2018 - Iris recognition in cases of eye pathology (Trokielewicz et al.) [(paper)](http://arxiv.org/pdf/1809.01040v1.pdf) 
* 2018 - Linear regression analysis of template aging in iris biometrics (Trokielewicz) [(paper)](http://arxiv.org/pdf/1809.00170v1.pdf) 
* 2018 - Robust Iris Segmentation Based on Fully Convolutional Networks and Generative Adversarial Networks (Bezerra et al.) [(paper)](http://arxiv.org/pdf/1809.00769v1.pdf) 
* 2018 - SegDenseNet: Iris Segmentation for Pre and Post Cataract Surgery (Lakra et al.) [(paper)](http://arxiv.org/pdf/1801.10100v2.pdf) 
* 2018 - The Impact of Preprocessing on Deep Representations for Iris Recognition on Unconstrained Environments (Zanlorensi et al.) [(paper)](http://arxiv.org/pdf/1808.10032v1.pdf) 
* 2018 - Unconstrained Iris Segmentation using Convolutional Neural Networks (Ahmad et al.) [(paper)](http://arxiv.org/pdf/1812.08245v1.pdf) 
* 2017 - An End to End Deep Neural Network for Iris Segmentation in Unconstraint Scenarios (Bazrafkan et al.) [(paper)](http://arxiv.org/pdf/1712.02877v1.pdf) 
* 2017 - An Experimental Study of Deep Convolutional Features For Iris Recognition (Minaee et al.) [(paper)](http://arxiv.org/pdf/1702.01334v1.pdf) 
* 2017 - GHCLNet: A Generalized Hierarchically tuned Contact Lens detection Network (Singh et al.) [(paper)](http://arxiv.org/pdf/1710.05152v1.pdf) 
* 2017 - Gender-From-Iris or Gender-From-Mascara? (Kuehlkamp et al.) [(paper)](http://arxiv.org/pdf/1702.01304v1.pdf) 
* 2017 - IRINA: Iris Recognition (Even) in Inaccurately Segmented Data (Proenca et al.) [(paper)](http://openaccess.thecvf.com/content_cvpr_2017/papers/Proenca_IRINA_Iris_Recognition_CVPR_2017_paper.pdf) 
* 2017 - Improving RANSAC-Based Segmentation Through CNN Encapsulation (Morley et al.) [(paper)](http://openaccess.thecvf.com/content_cvpr_2017/papers/Morley_Improving_RANSAC-Based_Segmentation_CVPR_2017_paper.pdf) 
* 2017 - Towards More Accurate Iris Recognition Using Deeply Learned Spatially Corresponding Features (Zhao et al.) [(paper)](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhao_Towards_More_Accurate_ICCV_2017_paper.pdf) 
* 2017 - UBSegNet: Unified Biometric Region of Interest Segmentation Network (Jha et al.) [(paper)](http://arxiv.org/pdf/1709.08924v1.pdf) 
* 2016 - Deep iris representation with applications in iris recognition and cross-sensor iris recognition (abhishek et al.) [(paper)](http://apps.webofknowledge.com/full_record.do?product=WOS&search_mode=GeneralSearch&qid=4&SID=8DMTkMpwgZDXBYO5ilS&page=1&doc=1) 
* 2015 - An Accurate Iris Segmentation Framework Under Relaxed Imaging Constraints Using Total Variation Model (Zhao et al.) [(paper)](http://openaccess.thecvf.com/content_iccv_2015/papers/Zhao_An_Accurate_Iris_ICCV_2015_paper.pdf) 
* 2015 - Iris Recognition Using Scattering Transform and Textural Features (Minaee et al.) [(paper)](http://arxiv.org/pdf/1507.02177v1.pdf) 
* 2011 - A New IRIS Normalization Process For Recognition System With Cryptographic Techniques (Nithyanandam et al.) [(paper)](http://arxiv.org/pdf/1111.5135v1.pdf) 
* 2011 - Exploring New Directions in Iris Recognition (Popescu-Bodorin) [(paper)](http://arxiv.org/pdf/1107.2696v1.pdf) 
* 2011 - Iris Recognition Based on LBP and Combined LVQ Classifier (Shams et al.) [(paper)](http://arxiv.org/pdf/1111.1562v1.pdf) 
<br/><br/>

---
<a name="macular-degeneration"></a>
### Macular Degeneration
---
* 2021 - AI Fairness via Domain Adaptation (Joshi et al.) [(paper)](https://arxiv.org/pdf/2104.01109v1.pdf) 
* 2021 - Analyzing Epistemic and Aleatoric Uncertainty for Drusen Segmentation in Optical Coherence Tomography Images (Joy et al.) [(paper)](https://arxiv.org/pdf/2101.08888v2.pdf) [(code)](https://github.com/ssedai026/uncertainty-segmentation)
* 2021 - Uncertainty aware and explainable diagnosis of retinal disease (Singh et al.) [(paper)](https://arxiv.org/pdf/2101.12041v1.pdf) 
* 2020 - Addressing Artificial Intelligence Bias in Retinal Disease Diagnostics (Burlina et al.) [(paper)](https://arxiv.org/pdf/2004.13515v4.pdf) 
* 2020 - Automated data extraction of bar chart raster images (Carderas et al.) [(paper)](https://arxiv.org/pdf/2011.04137v1.pdf) 
* 2020 - Automatic detection and counting of retina cell nuclei using deep learning (Hosseini et al.) [(paper)](https://arxiv.org/pdf/2002.03563v1.pdf) 
* 2020 - Fundus Image Analysis for Age Related Macular Degeneration: ADAM-2020 Challenge Report (Shankaranarayana) [(paper)](https://arxiv.org/pdf/2009.01548v1.pdf) 
* 2020 - Globally Optimal Surface Segmentation using Deep Learning with Learnable Smoothness Priors (Zhou et al.) [(paper)](https://arxiv.org/pdf/2007.01217v1.pdf) 
* 2020 - Predicting risk of late age-related macular degeneration using deep learning (Peng et al.) [(paper)](https://arxiv.org/pdf/2007.09550v1.pdf) 
* 2020 - Synergic Adversarial Label Learning for Grading Retinal Diseases via Knowledge Distillation and Multi-task Learning (Ju et al.) [(paper)](https://arxiv.org/pdf/2003.10607v4.pdf) 
* 2020 - The relationship between Fully Connected Layers and number of classes for the analysis of retinal images (Ram et al.) [(paper)](https://arxiv.org/pdf/2004.03624v2.pdf) 
* 2020 - Unsupervised deep learning for grading of age-related macular degeneration using retinal fundus images (Yellapragada et al.) [(paper)](https://arxiv.org/pdf/2010.11993v1.pdf) 
* 2019 - A Deep-learning Approach for Prognosis of Age-Related Macular Degeneration Disease using SD-OCT Imaging Biomarkers (Banerjee et al.) [(paper)](http://arxiv.org/pdf/1902.10700v1.pdf) 
* 2019 - AMD Severity Prediction And Explainability Using Image Registration And Deep Embedded Clustering (Mahapatra) [(paper)](https://arxiv.org/pdf/1907.03075v1.pdf) 
* 2019 - Evaluation of a deep learning system for the joint automated detection of diabetic retinopathy and age-related macular degeneration (González-Gonzalo et al.) [(paper)](http://arxiv.org/pdf/1903.09555v1.pdf) 
* 2019 - Generative Adversarial Networks Synthesize Realistic OCT Images of the Retina (Odaibo et al.) [(paper)](http://arxiv.org/pdf/1902.06676v1.pdf) 
* 2019 - Iterative augmentation of visual evidence for weakly-supervised lesion localization in deep interpretability frameworks (González-Gonzalo et al.) [(paper)](https://arxiv.org/pdf/1910.07373v1.pdf) 
* 2019 - Predicting Progression of Age-related Macular Degeneration from Fundus Images using Deep Learning (Babenko et al.) [(paper)](http://arxiv.org/pdf/1904.05478v1.pdf) 
* 2019 - Two-Stream CNN with Loose Pair Training for Multi-modal AMD Categorization (Wang et al.) [(paper)](https://arxiv.org/pdf/1907.12023v1.pdf) 
* 2019 - retina-VAE: Variationally Decoding the Spectrum of Macular Disease (Odaibo) [(paper)](https://arxiv.org/pdf/1907.05195v1.pdf) 
* 2018 - A Consolidated Approach to Convolutional Neural Networks and the Kolmogorov Complexity (Mekontchou) [(paper)](http://arxiv.org/pdf/1812.00888v1.pdf) 
* 2018 - A Consolidated Approach to Convolutional Neural Networks and the Kolmogorov Complexity (Yomba) [(paper)](http://arxiv.org/pdf/1812.00888v1.pdf) 
* 2018 - A multi-task deep learning model for the classification of Age-related Macular Degeneration (Chen et al.) [(paper)](http://arxiv.org/pdf/1812.00422v1.pdf) 
* 2018 - DeepSeeNet: A deep learning model for automated classification of patient-based age-related macular degeneration severity from color fundus photographs (Peng et al.) [(paper)](http://arxiv.org/pdf/1811.07492v2.pdf) 
* 2018 - Unsupervised Identification of Disease Marker Candidates in Retinal OCT Imaging Data (Seeböck et al.) [(paper)](http://arxiv.org/pdf/1810.13404v1.pdf) 
* 2017 - Simultaneous Multiple Surface Segmentation Using Deep Learning (Shah et al.) [(paper)](http://arxiv.org/pdf/1705.07142v1.pdf) 
* 2015 - Morphometric analyses of the visual pathways in macular degeneration (Hernowo et al.) [(paper)](http://arxiv.org/pdf/1501.05391v1.pdf) 
<br/><br/>

---
<a name="retinal-fundus-classification"></a>
### Retinal Fundus Classification
---
* 2021 - A Benchmark of Ocular Disease Intelligent Recognition: One Shot for Multi-disease Detection (li et al.) [(paper)](https://arxiv.org/pdf/2102.07978v1.pdf) 
* 2021 - A Deep Learning Approach for Diabetic Retinopathy detection using Transfer Learning (Ramchandre et al.) [(paper)](https://ieeexplore.ieee.org/abstract/document/9298201/) 
* 2021 - A systematic review of transfer learning based approaches for diabetic retinopathy detection (Oltu et al.) [(paper)](https://arxiv.org/pdf/2105.13793v1.pdf) 
* 2021 - An Interpretable Multiple-Instance Approach for the Detection of referable Diabetic Retinopathy from Fundus Images (Papadopoulos et al.) [(paper)](https://arxiv.org/pdf/2103.01702v1.pdf) 
* 2021 - Consistent Posterior Distributions under Vessel-Mixing: A Regularization for Cross-Domain Retinal Artery/Vein Classification (Li et al.) [(paper)](https://arxiv.org/pdf/2103.09097v1.pdf) 
* 2021 - Contextualized Keyword Representations for Multi-modal Retinal Image Captioning (Huang et al.) [(paper)](https://arxiv.org/pdf/2104.12471v1.pdf) 
* 2021 - Defending Medical Image Diagnostics against Privacy Attacks using Generative Methods (Paul et al.) [(paper)](https://arxiv.org/pdf/2103.03078v1.pdf) 
* 2021 - DiaRet: A browser-based application for the grading of Diabetic Retinopathy with Integrated Gradients (Patel et al.) [(paper)](https://arxiv.org/pdf/2103.08501v3.pdf) 
* 2021 - Efficient Screening of Diseased Eyes based on Fundus Autofluorescence Images using Support Vector Machine (Manne et al.) [(paper)](https://arxiv.org/pdf/2104.08519v1.pdf) 
* 2021 - Glaucoma detection beyond the optic disc: The importance of the peripapillary region using explainable deep learning (Hemelings et al.) [(paper)](https://arxiv.org/pdf/2103.11895v1.pdf) 
* 2021 - Improving Medical Image Classification with Label Noise Using Dual-uncertainty Estimation (Ju et al.) [(paper)](https://arxiv.org/pdf/2103.00528v2.pdf) 
* 2021 - Multi-Disease Detection in Retinal Imaging based on Ensembling Heterogeneous Deep Learning Models (Müller et al.) [(paper)](https://arxiv.org/pdf/2103.14660v1.pdf) [(code)](https://github.com/frankkramer-lab/riadd.aucmedi)
* 2021 - Multitasking Deep Learning Model for Detection of Five Stages of Diabetic Retinopathy (Majumder et al.) [(paper)](https://arxiv.org/pdf/2103.04207v1.pdf) 
* 2021 - Relational Subsets Knowledge Distillation for Long-tailed Retinal Diseases Recognition (Ju et al.) [(paper)](https://arxiv.org/pdf/2104.11057v1.pdf) 
* 2021 - Self-Adaptive Transfer Learning for Multicenter Glaucoma Classification in Fundus Retina Images (Bao et al.) [(paper)](https://arxiv.org/pdf/2105.03068v1.pdf) 
* 2021 - The Usability and Trustworthiness of Medical Eye Images (Diethei et al.) [(paper)](http://arxiv.org/pdf/2105.12651v1.pdf) 
* 2020 - A Dark and Bright Channel Prior Guided Deep Network for Retinal Image Quality Assessment (Xu et al.) [(paper)](https://arxiv.org/pdf/2010.13313v2.pdf) 
* 2020 - A Deep Retinal Image Quality Assessment Network with Salient Structure Priors (Xu et al.) [(paper)](https://arxiv.org/pdf/2012.15575v1.pdf) 
* 2020 - A Fast and Effective Method of Macula Automatic Detection for Retina Images (Jiang et al.) [(paper)](http://arxiv.org/pdf/2010.03122v1.pdf) 
* 2020 - ADINet: Attribute Driven Incremental Network for Retinal Image Classification (Meng et al.) [(paper)](http://openaccess.thecvf.com/content_CVPR_2020/papers/Meng_ADINet_Attribute_Driven_Incremental_Network_for_Retinal_Image_Classification_CVPR_2020_paper.pdf) 
* 2020 - Adversarial Attack Vulnerability of Medical Image Analysis Systems: Unexplored Factors (Bortsova et al.) [(paper)](https://arxiv.org/pdf/2006.06356v3.pdf) [(code)](https://github.com/Gerda92/adversarial_transfer_factors)
* 2020 - Adversarial Exposure Attack on Diabetic Retinopathy Imagery (Cheng et al.) [(paper)](https://arxiv.org/pdf/2009.09231v1.pdf) 
* 2020 - An Active Learning Method for Diabetic Retinopathy Classification with Uncertainty Quantification (Ahsan et al.) [(paper)](https://arxiv.org/pdf/2012.13325v2.pdf) 
* 2020 - Automated Detection of Microaneurysms in Color Fundus Images using Deep Learning with Different Preprocessing Approaches (Tavakoli et al.) [(paper)](http://arxiv.org/pdf/2004.09493v1.pdf) 
* 2020 - Automated Diabetic Retinopathy Grading using Deep Convolutional Neural Network (Chaturvedi et al.) [(paper)](https://arxiv.org/pdf/2004.06334v1.pdf) 
* 2020 - Automated Smartphone based System for Diagnosis of Diabetic Retinopathy (Hagos et al.) [(paper)](https://arxiv.org/pdf/2004.03408v1.pdf) 
* 2020 - Blended Multi-Modal Deep ConvNet Features for Diabetic Retinopathy Severity Prediction (Bodapati et al.) [(paper)](https://arxiv.org/pdf/2006.00197v1.pdf) 
* 2020 - Bridge the Domain Gap Between Ultra-wide-field and Traditional Fundus Images via Adversarial Domain Adaptation (Ju et al.) [(paper)](https://arxiv.org/pdf/2003.10042v2.pdf) 
* 2020 - Classification of Diabetic Retinopathy Using Unlabeled Data and Knowledge Distillation (Abbasi et al.) [(paper)](https://arxiv.org/pdf/2009.00982v1.pdf) 
* 2020 - Classification of Diabetic Retinopathy via Fundus Photography: Utilization of Deep Learning Approaches to Speed up Disease Detection (Zhuang et al.) [(paper)](https://arxiv.org/pdf/2007.09478v1.pdf) 
* 2020 - Combining Fine- and Coarse-Grained Classifiers for Diabetic Retinopathy Detection (Bajwa et al.) [(paper)](https://arxiv.org/pdf/2005.14308v1.pdf) 
* 2020 - Comparison Different Vessel Segmentation Methods in Automated Microaneurysms Detection in Retinal Images using Convolutional Neural Networks (Tavakoli et al.) [(paper)](http://arxiv.org/pdf/2005.09097v1.pdf) 
* 2020 - Conversion and Implementation of State-of-the-Art Deep Learning Algorithms for the Classification of Diabetic Retinopathy (Rao et al.) [(paper)](https://arxiv.org/pdf/2010.11692v1.pdf) 
* 2020 - Cost-Sensitive Regularization for Diabetic Retinopathy Grading from Eye Fundus Images (Galdran et al.) [(paper)](https://arxiv.org/pdf/2010.00291v1.pdf) [(code)](https://github.com/agaldran/cost_sensitive_loss_classification)
* 2020 - DRDr II: Detecting the Severity Level of Diabetic Retinopathy Using Mask RCNN and Transfer Learning (Shenavarmasouleh et al.) [(paper)](https://arxiv.org/pdf/2011.14733v1.pdf) 
* 2020 - Deep Learning Approach to Diabetic Retinopathy Detection (Tymchenko et al.) [(paper)](https://arxiv.org/pdf/2003.02261v1.pdf) [(code)](https://github.com/debayanmitra1993-data/Blindness-Detection-Diabetic-Retinopathy-)
* 2020 - Detection of Diabetic Anomalies in Retinal Images using Morphological Cascading Decision Tree (Ghaffar et al.) [(paper)](https://arxiv.org/pdf/2001.01953v1.pdf) 
* 2020 - Diabetic Retinopathy Diagnosis based on Convolutional Neural Network (abed et al.) [(paper)](https://arxiv.org/pdf/2008.00148v1.pdf) 
* 2020 - Diabetic Retinopathy Grading System Based on Transfer Learning (AbdelMaksoud et al.) [(paper)](https://arxiv.org/pdf/2012.12515v1.pdf) 
* 2020 - Diabetic Retinopathy detection by retinal image recognizing (Junior) [(paper)](https://arxiv.org/pdf/2001.05835v1.pdf) 
* 2020 - Diagnosis of Diabetic Retinopathy in Ethiopia: Before the Deep Learning based Automation (Hagos) [(paper)](https://arxiv.org/pdf/2003.09208v2.pdf) 
* 2020 - Distractor-Aware Neuron Intrinsic Learning for Generic 2D Medical Image Classifications (Gong et al.) [(paper)](https://arxiv.org/pdf/2007.09979v2.pdf) 
* 2020 - Early Blindness Detection Based on Retinal Images Using Ensemble Learning (Sikder et al.) [(paper)](https://arxiv.org/pdf/2006.07475v1.pdf) 
* 2020 - Early Detection of Retinopathy of Prematurity (ROP) in Retinal Fundus Images Via Convolutional Neural Networks (Guo et al.) [(paper)](https://arxiv.org/pdf/2006.06968v1.pdf) 
* 2020 - Evaluating Knowledge Transfer in Neural Network for Medical Images (Akbarian et al.) [(paper)](https://arxiv.org/pdf/2008.13574v2.pdf) 
* 2020 - Explainable end-to-end deep learning for diabetic retinopathy detection across multiple datasets (Chetoui et al.) [(paper)](https://www.spiedigitallibrary.org/journalArticle/Download?fullDOI=10.1117%2F1.JMI.7.4.044503) 
* 2020 - Exploiting Uncertainties from Ensemble Learners to Improve Decision-Making in Healthcare AI (Tan et al.) [(paper)](https://arxiv.org/pdf/2007.06063v1.pdf) 
* 2020 - GREEN: a Graph REsidual rE-ranking Network for Grading Diabetic Retinopathy (Liu et al.) [(paper)](https://arxiv.org/pdf/2007.09968v2.pdf) 
* 2020 - Hybrid Deep Learning Gaussian Process for Diabetic Retinopathy Diagnosis and Uncertainty Quantification (Toledo-Cortés et al.) [(paper)](https://arxiv.org/pdf/2007.14994v1.pdf) [(code)](https://github.com/FernandoTakenaka-Unifei/PCO102)
* 2020 - Improving Medical Annotation Quality to Decrease Labeling Burden Using Stratified Noisy Cross-Validation (Hsu et al.) [(paper)](https://arxiv.org/pdf/2009.10858v1.pdf) 
* 2020 - Kernel Self-Attention in Deep Multiple Instance Learning (Rymarczyk et al.) [(paper)](https://arxiv.org/pdf/2005.12991v2.pdf) 
* 2020 - Learning Discriminative Representations for Fine-Grained Diabetic Retinopathy Grading (Tian et al.) [(paper)](https://arxiv.org/pdf/2011.02120v1.pdf) 
* 2020 - Multi-modal, multi-task, multi-attention (M3) deep learning detection of reticular pseudodrusen: towards automated and accessible classification of age-related macular degeneration (Chen et al.) [(paper)](https://arxiv.org/pdf/2011.05142v2.pdf) 
* 2020 - Optic disc and fovea localisation in ultra-widefield scanning laser ophthalmoscope images captured in multiple modalities (Wakeford et al.) [(paper)](http://arxiv.org/pdf/2004.11691v1.pdf) 
* 2020 - Optimal Transfer Learning Model for Binary Classification of Funduscopic Images through Simple Heuristics (Jammula et al.) [(paper)](https://arxiv.org/pdf/2002.04189v3.pdf) 
* 2020 - Point-of-Care Diabetic Retinopathy Diagnosis: A Standalone Mobile Application Approach (Hagos) [(paper)](https://arxiv.org/pdf/2002.04066v1.pdf) 
* 2020 - Predicting Risk of Developing Diabetic Retinopathy using Deep Learning (Bora et al.) [(paper)](https://arxiv.org/pdf/2008.04370v1.pdf) 
* 2020 - Predictive Analysis of Diabetic Retinopathy with Transfer Learning (Labhsetwar et al.) [(paper)](https://arxiv.org/pdf/2011.04052v2.pdf) 
* 2020 - Pseudo-Labeling for Small Lesion Detection on Diabetic Retinopathy Images (Chen et al.) [(paper)](https://arxiv.org/pdf/2003.12040v1.pdf) 
* 2020 - Residual-CycleGAN based Camera Adaptation for Robust Diabetic Retinopathy Screening (Yang et al.) [(paper)](https://arxiv.org/pdf/2007.15874v1.pdf) 
* 2020 - Robust Collaborative Learning of Patch-level and Image-level Annotations for Diabetic Retinopathy Grading from Fundus Image (Yang et al.) [(paper)](https://arxiv.org/pdf/2008.00610v2.pdf) [(code)](https://github.com/PaddlePaddle/Research)
* 2020 - Self-supervised Feature Learning via Exploiting Multi-modal Data for Retinal Disease Diagnosis (Li et al.) [(paper)](https://arxiv.org/pdf/2007.11067v1.pdf) [(code)](https://github.com/xmengli999/self_supervised)
* 2020 - Smartphone-Based Test and Predictive Models for Rapid, Non-Invasive, and Point-of-Care Monitoring of Ocular and Cardiovascular Complications Related to Diabetes (Chakravadhanula) [(paper)](https://arxiv.org/pdf/2011.08068v1.pdf) 
* 2020 - TR-GAN: Topology Ranking GAN with Triplet Loss for Retinal Artery/Vein Classification (Chen et al.) [(paper)](https://arxiv.org/pdf/2007.14852v1.pdf) 
* 2020 - The Efficacy of Microaneurysms Detection With and Without Vessel Segmentation in Color Retinal Images (Tavakoli et al.) [(paper)](http://arxiv.org/pdf/2005.09098v1.pdf) 
* 2020 - Towards the Localisation of Lesions in Diabetic Retinopathy (Mensah et al.) [(paper)](https://arxiv.org/pdf/2012.11432v2.pdf) 
* 2020 - Two-stage framework for optic disc localization and glaucoma classification in retinal fundus images using deep learning (Bajwa et al.) [(paper)](https://arxiv.org/pdf/2005.14284v1.pdf) 
* 2019 - A Deep Step Pattern Representation for Multimodal Retinal Image Registration (Lee et al.) [(paper)](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_A_Deep_Step_Pattern_Representation_for_Multimodal_Retinal_Image_Registration_ICCV_2019_paper.pdf) 
* 2019 - A Novel Deep Learning Pipeline for Retinal Vessel Detection in Fluorescein Angiography (Ding et al.) [(paper)](https://arxiv.org/pdf/1907.02946v2.pdf) 
* 2019 - A complementary method for automated detection of microaneurysms in fluorescein angiography fundus images to assess diabetic retinopathy (Tavakoli et al.) [(paper)](http://arxiv.org/pdf/1909.01557v1.pdf) 
* 2019 - A deep learning approach for automated detection of geographic atrophy from color fundus photographs (Keenan et al.) [(paper)](https://arxiv.org/pdf/1906.03153v1.pdf) [(code)](https://github.com/ncbi-nlp/DeepSeeNet)
* 2019 - Advances in Computer-Aided Diagnosis of Diabetic Retinopathy (Chaturvedi et al.) [(paper)](https://arxiv.org/pdf/1909.09853v1.pdf) 
* 2019 - Artificial Intelligence for Pediatric Ophthalmology (Reid et al.) [(paper)](http://arxiv.org/pdf/1904.08796v1.pdf) 
* 2019 - Assessment of central serous chorioretinopathy (CSC) depicted on color fundus photographs using deep Learning (Zhen et al.) [(paper)](http://arxiv.org/pdf/1901.04540v1.pdf) 
* 2019 - Attention Based Glaucoma Detection: A Large-scale Database and CNN Model (Li et al.) [(paper)](http://arxiv.org/pdf/1903.10831v3.pdf) [(code)](https://github.com/smilell/AG-CNN)
* 2019 - Automatic detection of rare pathologies in fundus photographs using few-shot learning (Quellec et al.) [(paper)](https://arxiv.org/pdf/1907.09449v3.pdf) 
* 2019 - CANet: Cross-disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading (Li et al.) [(paper)](https://arxiv.org/pdf/1911.01376v1.pdf) [(code)](https://github.com/xmengli999/CANet)
* 2019 - CaRENets: Compact and Resource-Efficient CNN for Homomorphic Inference on Encrypted Medical Images (Chao et al.) [(paper)](http://arxiv.org/pdf/1901.10074v1.pdf) 
* 2019 - Computationally Efficient Optic Nerve Head Detection in Retinal Fundus Images (Pourreza-Shahri et al.) [(paper)](http://arxiv.org/pdf/1909.01558v2.pdf) 
* 2019 - DR$\vert$GRADUATE: uncertainty-aware deep learning-based diabetic retinopathy grading in eye fundus images (Araújo et al.) [(paper)](https://arxiv.org/pdf/1910.11777v2.pdf) 
* 2019 - Deep Learning Fundus Image Analysis for Diabetic Retinopathy and Macular Edema Grading (Sahlsten et al.) [(paper)](http://arxiv.org/pdf/1904.08764v1.pdf) 
* 2019 - Detecting Anemia from Retinal Fundus Images (Mitani et al.) [(paper)](http://arxiv.org/pdf/1904.06435v1.pdf) 
* 2019 - Direct Classification of Type 2 Diabetes From Retinal Fundus Images in a Population-based Sample From The Maastricht Study (Heslinga et al.) [(paper)](https://arxiv.org/pdf/1911.10022v1.pdf) 
* 2019 - Early Detection of Diabetic Retinopathy and Severity Scale Measurement: A Progressive Review & Scopes (Khatun et al.) [(paper)](https://arxiv.org/pdf/1912.12829v1.pdf) 
* 2019 - Edge, Ridge, and Blob Detection with Symmetric Molecules (Reisenhofer et al.) [(paper)](https://arxiv.org/pdf/1901.09723v2.pdf) 
* 2019 - Evaluation of Retinal Image Quality Assessment Networks in Different Color-spaces (Fu et al.) [(paper)](https://arxiv.org/pdf/1907.05345v4.pdf) [(code)](https://github.com/HzFu/EyeQ_Enhancement)
* 2019 - Evaluation of an AI System for the Detection of Diabetic Retinopathy from Images Captured with a Handheld Portable Fundus Camera: the MAILOR AI study (Rogers et al.) [(paper)](https://arxiv.org/pdf/1908.06399v1.pdf) 
* 2019 - GLAMpoints: Greedily Learned Accurate Match points (Truong et al.) [(paper)](https://arxiv.org/pdf/1908.06812v3.pdf) [(code)](https://github.com/PruneTruong/GLAMpoints_pytorch)
* 2019 - Gaze-Contingent Ocular Parallax Rendering for Virtual Reality (Konrad et al.) [(paper)](http://arxiv.org/pdf/1906.09740v2.pdf) 
* 2019 - Hierarchical method for cataract grading based on retinal images using improved Haar wavelet (Cao et al.) [(paper)](http://arxiv.org/pdf/1904.01261v1.pdf) 
* 2019 - Instant automatic diagnosis of diabetic retinopathy (Quellec et al.) [(paper)](https://arxiv.org/pdf/1906.11875v1.pdf) 
* 2019 - Mini Lesions Detection on Diabetic Retinopathy Images via Large Scale CNN Features (Chen et al.) [(paper)](https://arxiv.org/pdf/1911.08588v1.pdf) 
* 2019 - O-MedAL: Online Active Deep Learning for Medical Image Analysis (Smailagic et al.) [(paper)](https://arxiv.org/pdf/1908.10508v2.pdf) [(code)](https://github.com/adgaudio/O-MedAL)
* 2019 - Reproduction study using public data of: Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs (Voets et al.) [(paper)](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0217541&type=printable) [(code)](https://github.com/mikevoets/jama16-retina-replication)
* 2019 - Transfer Learning based Detection of Diabetic Retinopathy from Small Dataset (Hagos et al.) [(paper)](https://arxiv.org/pdf/1905.07203v2.pdf) [(code)](https://github.com/ShubhayanS/Multiclass-Diabetic-Retinopathy-Detection)
* 2018 - A Multi-task Network to Detect Junctions in Retinal Vasculature (Uslu et al.) [(paper)](http://arxiv.org/pdf/1806.03175v1.pdf) 
* 2018 - Capsule Networks against Medical Imaging Data Challenges (Jiménez-Sánchez et al.) [(paper)](http://arxiv.org/pdf/1807.07559v1.pdf) 
* 2018 - Classification of Findings with Localized Lesions in Fundoscopic Images using a Regionally Guided CNN (Son et al.) [(paper)](http://arxiv.org/pdf/1811.00871v1.pdf) 
* 2018 - Deep Learning based Early Detection and Grading of Diabetic Retinopathy Using Retinal Fundus Images (Islam et al.) [(paper)](http://arxiv.org/pdf/1812.10595v1.pdf) [(code)](https://github.com/saifulislampharma/ratinopathy)
* 2018 - Deep Learning vs. Human Graders for Classifying Severity Levels of Diabetic Retinopathy in a Real-World Nationwide Screening Program (Raumviboonsuk et al.) [(paper)](http://arxiv.org/pdf/1810.08290v1.pdf) 
* 2018 - Detection of Hard Exudates in Retinal Fundus Images using Deep Learning (Benzamin et al.) [(paper)](http://arxiv.org/pdf/1808.03656v1.pdf) 
* 2018 - Ensemble of Convolutional Neural Networks for Automatic Grading of Diabetic Retinopathy and Macular Edema (Kori et al.) [(paper)](http://arxiv.org/pdf/1809.04228v1.pdf) 
* 2018 - Identification and Visualization of the Underlying Independent Causes of the Diagnostic of Diabetic Retinopathy made by a Deep Learning Classifier (Torre et al.) [(paper)](http://arxiv.org/pdf/1809.08567v1.pdf) 
* 2018 - Multi-Cell Multi-Task Convolutional Neural Networks for Diabetic Retinopathy Grading (Zhou et al.) [(paper)](http://arxiv.org/pdf/1808.10564v2.pdf) 
* 2018 - Multimodal Registration of Retinal Images Using Domain-Specific Landmarks and Vessel Enhancement (Hervella et al.) [(paper)](http://arxiv.org/pdf/1803.00951v2.pdf) 
* 2018 - Pathological Evidence Exploration in Deep Retinal Image Diagnosis (Niu et al.) [(paper)](http://arxiv.org/pdf/1812.02640v1.pdf) 
* 2018 - Predicting optical coherence tomography-derived diabetic macular edema grades from fundus photographs using deep learning (Varadarajan et al.) [(paper)](https://arxiv.org/pdf/1810.10342v4.pdf) 
* 2018 - Relation Networks for Optic Disc and Fovea Localization in Retinal Images (Chandra et al.) [(paper)](http://arxiv.org/pdf/1812.00883v1.pdf) 
* 2018 - Relation Networks for Optic Disc and Fovea Localization in Retinal Images (Babu et al.) [(paper)](http://arxiv.org/pdf/1812.00883v1.pdf) 
* 2018 - Replication study: Development and validation of deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs (Voets et al.) [(paper)](http://arxiv.org/pdf/1803.04337v3.pdf) [(code)](https://github.com/mikevoets/jama16-retina-replication)
* 2017 - An Ensemble Deep Learning Based Approach for Red Lesion Detection in Fundus Images (Orlando et al.) [(paper)](http://arxiv.org/pdf/1706.03008v2.pdf) [(code)](https://github.com/ignaciorlando/red-lesion-detection)
* 2017 - Automatic Classification of Bright Retinal Lesions via Deep Network Features (Sadek et al.) [(paper)](http://arxiv.org/pdf/1707.02022v3.pdf) [(code)](https://github.com/mawady/DeepRetinalClassification)
* 2017 - Classification of Diabetic Retinopathy Images Using Multi-Class Multiple-Instance Learning Based on Color Correlogram Features (Venkatesan et al.) [(paper)](http://arxiv.org/pdf/1704.01264v1.pdf) 
* 2017 - Deep Learning for Automated Quality Assessment of Color Fundus Images in Diabetic Retinopathy Screening (Saha et al.) [(paper)](http://arxiv.org/pdf/1703.02511v1.pdf) 
* 2017 - Deep learning for predicting refractive error from retinal fundus images (Varadarajan et al.) [(paper)](http://arxiv.org/pdf/1712.07798v1.pdf) 
* 2017 - Diabetic Retinopathy Detection via Deep Convolutional Networks for Discriminative Localization and Visual Explanation (Wang et al.) [(paper)](https://arxiv.org/pdf/1703.10757v3.pdf) [(code)](https://github.com/Ram-Aditya/Healthcare-Data-Analytics)
* 2017 - Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy (Krause et al.) [(paper)](http://arxiv.org/pdf/1710.01711v3.pdf) 
* 2017 - Institutionally Distributed Deep Learning Networks (Chang et al.) [(paper)](http://arxiv.org/pdf/1709.05929v1.pdf) 
* 2017 - Learning Deep Representations of Medical Images using Siamese CNNs with Application to Content-Based Image Retrieval (Chung et al.) [(paper)](http://arxiv.org/pdf/1711.08490v2.pdf) 
* 2017 - Lesion detection and Grading of Diabetic Retinopathy via Two-stages Deep Convolutional Neural Networks (Yang et al.) [(paper)](http://arxiv.org/pdf/1705.00771v1.pdf) 
* 2017 - Microaneurysm Detection in Fundus Images Using a Two-step Convolutional Neural Networks (Eftekheri et al.) [(paper)](http://arxiv.org/pdf/1710.05191v2.pdf) 
* 2017 - Predicting Cardiovascular Risk Factors from Retinal Fundus Photographs using Deep Learning (Poplin et al.) [(paper)](http://arxiv.org/pdf/1708.09843v2.pdf) 
* 2017 - Retinal Microaneurysms Detection using Local Convergence Index Features (Dashtbozorg et al.) [(paper)](http://arxiv.org/pdf/1707.06865v1.pdf) [(code)](https://github.com/ChibingXiang/Local-Convergence-Index-Features)
* 2017 - Synthesising Wider Field Images from Narrow-Field Retinal Video Acquired Using a Low-Cost Direct Ophthalmoscope (Arclight) Attached to a Smartphone (Viquez et al.) [(paper)](http://arxiv.org/pdf/1708.07977v1.pdf) 
* 2017 - Weakly-supervised localization of diabetic retinopathy lesions in retinal fundus images (Gondal et al.) [(paper)](http://arxiv.org/pdf/1706.09634v1.pdf) 
* 2017 - Who Said What: Modeling Individual Labelers Improves Classification (Guan et al.) [(paper)](http://arxiv.org/pdf/1703.08774v2.pdf) [(code)](https://github.com/seunghyukcho/doctornet-pytorch)
* 2017 - Zoom-in-Net: Deep Mining Lesions for Diabetic Retinopathy Detection (Wang et al.) [(paper)](http://arxiv.org/pdf/1706.04372v1.pdf) 
* 2016 - A generalized flow for multi-class and binary classification tasks: An Azure ML approach (Bihis et al.) [(paper)](http://arxiv.org/pdf/1603.08070v1.pdf) 
* 2016 - An Unsupervised Method for Detection and Validation of The Optic Disc and The Fovea (Haloi et al.) [(paper)](http://arxiv.org/pdf/1601.06608v1.pdf) 
* 2016 - Automatic Discrimination of Color Retinal Images using the Bag of Words Approach (Sadek) [(paper)](http://arxiv.org/pdf/1603.04327v1.pdf) 
* 2016 - Automatic Identification of Retinal Arteries and Veins in Fundus Images using Local Binary Patterns (Hatami et al.) [(paper)](http://arxiv.org/pdf/1605.00763v1.pdf) 
* 2016 - Classification of Large-Scale Fundus Image Data Sets: A Cloud-Computing Framework (Roychowdhury) [(paper)](http://arxiv.org/pdf/1603.08071v1.pdf) 
* 2016 - Deep image mining for diabetic retinopathy screening (Quellec et al.) [(paper)](http://arxiv.org/pdf/1610.07086v3.pdf) 
* 2016 - Ensemble of Deep Convolutional Neural Networks for Learning to Detect Retinal Vessels in Fundus Images (Maji et al.) [(paper)](http://arxiv.org/pdf/1603.04833v1.pdf) [(code)](https://github.com/qiaotian/VesselSeg)
* 2016 - Neural Networks with Manifold Learning for Diabetic Retinopathy Detection (Rajanna et al.) [(paper)](http://arxiv.org/pdf/1612.03961v1.pdf) 
* 2016 - Superimposition of eye fundus images for longitudinal analysis from large public health databases (Noyel et al.) [(paper)](http://arxiv.org/pdf/1607.01971v3.pdf) 
* 2016 - Template Matching via Densities on the Roto-Translation Group (Bekkers et al.) [(paper)](http://arxiv.org/pdf/1603.03304v5.pdf) 
* 2016 - Tracking of Lines in Spherical Images via Sub-Riemannian Geodesics on SO(3) (Mashtakov et al.) [(paper)](http://arxiv.org/pdf/1604.03800v2.pdf) 
* 2015 - A Gaussian Scale Space Approach For Exudates Detection, Classification And Severity Prediction (Haloi et al.) [(paper)](http://arxiv.org/pdf/1505.00737v1.pdf) 
* 2015 - A Low-Dimensional Step Pattern Analysis Algorithm With Application to Multimodal Retinal Image Registration (Lee et al.) [(paper)](http://openaccess.thecvf.com/content_cvpr_2015/papers/Lee_A_Low-Dimensional_Step_2015_CVPR_paper.pdf) 
* 2015 - Improved Microaneurysm Detection using Deep Neural Networks (Haloi) [(paper)](http://arxiv.org/pdf/1505.04424v2.pdf) 
* 2015 - Performance Analysis of Cone Detection Algorithms (Mariotti et al.) [(paper)](http://arxiv.org/pdf/1502.01643v1.pdf) 
* 2015 - Simpler Non-Parametric Methods Provide as Good or Better Results to Multiple-Instance Learning (Venkatesan et al.) [(paper)](http://openaccess.thecvf.com/content_iccv_2015/papers/Venkatesan_Simpler_Non-Parametric_Methods_ICCV_2015_paper.pdf) 
* 2015 - Simpler non-parametric methods provide as good or better results to multiple-instance learning. (Venkatesan et al.) [(paper)](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Venkatesan_Simpler_Non-Parametric_Methods_ICCV_2015_paper.pdf) [(code)](https://github.com/ragavvenkatesan/np-mil)
* 2014 - A Two-phase Decision Support Framework for the Automatic Screening of Digital Fundus Images (Antal et al.) [(paper)](http://arxiv.org/pdf/1411.0130v1.pdf) 
* 2014 - An Ensemble-based System for Microaneurysm Detection and Diabetic Retinopathy Grading (Antal et al.) [(paper)](http://arxiv.org/pdf/1410.8577v1.pdf) 
* 2014 - An ensemble-based system for automatic screening of diabetic retinopathy (Antal et al.) [(paper)](http://arxiv.org/pdf/1410.8576v1.pdf) 
* 2013 - Improvement of Automatic Hemorrhages Detection Methods Using Shapes Recognition (Abbadi et al.) [(paper)](http://arxiv.org/pdf/1310.5999v1.pdf) 
* 2012 - A Multi-Orientation Analysis Approach to Retinal Vessel Tracking (Bekkers et al.) [(paper)](http://arxiv.org/pdf/1212.3530v5.pdf) 
* 2012 - A Session Based Blind Watermarking Technique within the NROI of Retinal Fundus Images for Authentication Using DWT, Spread Spectrum and Harris Corner Detection (Dey et al.) [(paper)](http://arxiv.org/pdf/1209.0053v1.pdf) 
* 2010 - Automatic diagnosis of retinal diseases from color retinal images (Jayanthi et al.) [(paper)](http://arxiv.org/pdf/1002.2408v1.pdf) 
<br/><br/>

---
<a name="retinal-fundus-segmentation"></a>
### Retinal Fundus Segmentation
---
* 2021 - AOSLO-net: A deep learning-based method for automatic segmentation of retinal microaneurysms from adaptive optics scanning laser ophthalmoscope images (Zhang et al.) [(paper)](https://arxiv.org/pdf/2106.02800v1.pdf) 
* 2021 - Advances in Classifying the Stages of Diabetic Retinopathy Using Convolutional Neural Networks in Low Memory Edge Devices (Paul) [(paper)](https://arxiv.org/pdf/2106.01739v1.pdf) 
* 2021 - BEFD: Boundary Enhancement and Feature Denoising for Vessel Segmentation (Zhang et al.) [(paper)](https://arxiv.org/pdf/2104.03768v1.pdf) 
* 2021 - Contextual Information Enhanced Convolutional Neural Networks for Retinal Vessel Segmentation in Color Fundus Images (Sun et al.) [(paper)](https://arxiv.org/pdf/2103.13622v1.pdf) 
* 2021 - Early Prediction and Diagnosis of Retinoblastoma Using Deep Learning Techniques (Durai et al.) [(paper)](https://arxiv.org/pdf/2103.07622v1.pdf) 
* 2021 - Exploring The Limits Of Data Augmentation For Retinal Vessel Segmentation (Uysal et al.) [(paper)](https://arxiv.org/pdf/2105.09365v2.pdf) [(code)](https://github.com/onurboyar/Retinal-Vessel-Segmentation)
* 2021 - Hierarchical Deep Network with Uncertainty-aware Semi-supervised Learning for Vessel Segmentation (Li et al.) [(paper)](https://arxiv.org/pdf/2105.14732v1.pdf) 
* 2021 - Learning to Address Intra-segment Misclassification in Retinal Imaging (Zhou et al.) [(paper)](https://arxiv.org/pdf/2104.12138v1.pdf) [(code)](https://github.com/rmaphoh/Learning_AVSegmentation)
* 2021 - M-Net with Bidirectional ConvLSTM for Cup and Disc Segmentation in Fundus Images (Khan et al.) [(paper)](https://arxiv.org/pdf/2104.03549v1.pdf) 
* 2021 - Objective-Dependent Uncertainty Driven Retinal Vessel Segmentation (Mishra et al.) [(paper)](https://arxiv.org/pdf/2104.08554v1.pdf) 
* 2021 - Pyramid U-Net for Retinal Vessel Segmentation (Zhang et al.) [(paper)](https://arxiv.org/pdf/2104.02333v1.pdf) 
* 2021 - RV-GAN: Segmenting Retinal Vascular Structure in Fundus Photographs using a Novel Multi-scale Generative Adversarial Network (Kamran et al.) [(paper)](https://arxiv.org/pdf/2101.00535v2.pdf) [(code)](https://github.com/SharifAmit/RVGAN)
* 2021 - SS-CADA: A Semi-Supervised Cross-Anatomy Domain Adaptation for Coronary Artery Segmentation (Zhang et al.) [(paper)](https://arxiv.org/pdf/2105.02674v1.pdf) 
* 2021 - Study Group Learning: Improving Retinal Vessel Segmentation Trained with Noisy Labels (Zhou et al.) [(paper)](https://arxiv.org/pdf/2103.03451v1.pdf) [(code)](https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation)
* 2020 - 3D Self-Supervised Methods for Medical Imaging (Taleb et al.) [(paper)](https://arxiv.org/pdf/2006.03829v3.pdf) [(code)](https://github.com/HealthML/self-supervised-3d-tasks)
* 2020 - A Benchmark for Studying Diabetic Retinopathy: Segmentation, Grading, and Transferability (Zhou et al.) [(paper)](https://arxiv.org/pdf/2008.09772v3.pdf) 
* 2020 - A Two-Stream Meticulous Processing Network for Retinal Vessel Segmentation (Zheng et al.) [(paper)](https://arxiv.org/pdf/2001.05829v1.pdf) 
* 2020 - An Efficient Framework for Automated Screening of Clinically Significant Macular Edema (Chalakkal et al.) [(paper)](http://arxiv.org/pdf/2001.07002v1.pdf) 
* 2020 - Automated Fovea Detection Based on Unsupervised Retinal Vessel Segmentation Method (Tavakoli et al.) [(paper)](http://arxiv.org/pdf/2004.08540v1.pdf) 
* 2020 - Automated Optic Nerve Head Detection Based on Different Retinal Vasculature Segmentation Methods and Mathematical Morphology (Tavakoli et al.) [(paper)](http://arxiv.org/pdf/2004.10253v1.pdf) 
* 2020 - Automatic lesion segmentation and Pathological Myopia classification in fundus images (Freire et al.) [(paper)](https://arxiv.org/pdf/2002.06382v1.pdf) 
* 2020 - Channel Attention Residual U-Net for Retinal Vessel Segmentation (Guo et al.) [(paper)](https://arxiv.org/pdf/2004.03702v5.pdf) [(code)](https://github.com/clguo/CAR-UNet)
* 2020 - Comparing Different Preprocessing Methods in Automated Segmentation of Retinal Vasculature (Tavakoli et al.) [(paper)](http://arxiv.org/pdf/2004.11696v1.pdf) 
* 2020 - Convex Shape Prior for Deep Neural Convolution Network based Eye Fundus Images Segmentation (Liu et al.) [(paper)](https://arxiv.org/pdf/2005.07476v1.pdf) 
* 2020 - DPN: Detail-Preserving Network with High Resolution Representation for Efficient Segmentation of Retinal Vessels (Guo) [(paper)](https://arxiv.org/pdf/2009.12053v1.pdf) [(code)](https://github.com/guomugong/DPN)
* 2020 - DRDr: Automatic Masking of Exudates and Microaneurysms Caused By Diabetic Retinopathy Using Mask R-CNN and Transfer Learning (Shenavarmasouleh et al.) [(paper)](https://arxiv.org/pdf/2007.02026v1.pdf) 
* 2020 - DeepOpht: Medical Report Generation for Retinal Images via Deep Models and Visual Explanation (Huang et al.) [(paper)](https://arxiv.org/pdf/2011.00569v1.pdf) [(code)](https://github.com/Jhhuangkay/DeepOpht-Medical-Report-Generation-for-Retinal-Images-via-Deep-Models-and-Visual-Explanation)
* 2020 - Dense Residual Network for Retinal Vessel Segmentation (Guo et al.) [(paper)](https://arxiv.org/pdf/2004.03697v1.pdf) [(code)](https://github.com/clguo/DRNet_Keras)
* 2020 - Detecting hidden signs of diabetes in external eye photographs (Babenko et al.) [(paper)](https://arxiv.org/pdf/2011.11732v1.pdf) 
* 2020 - Detection of Retinal Blood Vessels by using Gabor filter with Entropic threshold (Waly et al.) [(paper)](https://arxiv.org/pdf/2008.11508v1.pdf) 
* 2020 - Device for ECG prediction based on retinal vasculature analysis (Rai et al.) [(paper)](http://arxiv.org/pdf/2009.11099v1.pdf) 
* 2020 - Efficient Kernel based Matched Filter Approach for Segmentation of Retinal Blood Vessels (Saroj et al.) [(paper)](https://arxiv.org/pdf/2012.03601v1.pdf) 
* 2020 - Enhancement of Retinal Fundus Images via Pixel Color Amplification (Gaudio et al.) [(paper)](https://arxiv.org/pdf/2007.14456v1.pdf) [(code)](https://github.com/adgaudio/ietk-ret)
* 2020 - Evolutionary Neural Architecture Search for Retinal Vessel Segmentation (Fan et al.) [(paper)](https://arxiv.org/pdf/2001.06678v3.pdf) 
* 2020 - G1020: A Benchmark Retinal Fundus Image Dataset for Computer-Aided Glaucoma Detection (Bajwa et al.) [(paper)](https://arxiv.org/pdf/2006.09158v1.pdf) [(code)](https://github.com/mohaEs/G1020-segmentation-mask-generator)
* 2020 - Genetic U-Net: Automatically Designed Deep Networks for Retinal Vessel Segmentation Using a Genetic Algorithm (Wei et al.) [(paper)](https://arxiv.org/pdf/2010.15560v4.pdf) 
* 2020 - Grading the Severity of Arteriolosclerosis from Retinal Arterio-venous Crossing Patterns (Li et al.) [(paper)](https://arxiv.org/pdf/2011.03772v1.pdf) [(code)](https://github.com/conscienceli/MDTNet)
* 2020 - Improving Lesion Segmentation for Diabetic Retinopathy using Adversarial Learning (Xiao et al.) [(paper)](https://arxiv.org/pdf/2007.13854v1.pdf) [(code)](https://github.com/zoujx96/DR-segmentation)
* 2020 - Joint Learning of Vessel Segmentation and Artery/Vein Classification with Post-processing (Li et al.) [(paper)](https://arxiv.org/pdf/2005.13337v1.pdf) [(code)](https://github.com/conscienceli/SeqNet)
* 2020 - Learned Pre-Processing for Automatic Diabetic Retinopathy Detection on Eye Fundus Images (Smailagic et al.) [(paper)](https://arxiv.org/pdf/2007.13838v1.pdf) 
* 2020 - Leveraging Regular Fundus Images for Training UWF Fundus Diagnosis Models via Adversarial Learning and Pseudo-Labeling (Ju et al.) [(paper)](https://arxiv.org/pdf/2011.13816v2.pdf) 
* 2020 - Modeling and Enhancing Low-quality Retinal Fundus Images (Shen et al.) [(paper)](https://arxiv.org/pdf/2005.05594v3.pdf) [(code)](https://github.com/HzFu/EyeQ_Enhancement)
* 2020 - Monocular Retinal Depth Estimation and Joint Optic Disc and Cup Segmentation using Adversarial Networks (Shankaranarayana et al.) [(paper)](https://arxiv.org/pdf/2007.07502v1.pdf) 
* 2020 - Multi-Task Neural Networks with Spatial Activation for Retinal Vessel Segmentation and Artery/Vein Classification (Ma et al.) [(paper)](https://arxiv.org/pdf/2007.09337v1.pdf) 
* 2020 - Multimodal Transfer Learning-based Approaches for Retinal Vascular Segmentation (Morano et al.) [(paper)](https://arxiv.org/pdf/2012.10160v1.pdf) 
* 2020 - NuI-Go: Recursive Non-Local Encoder-Decoder Network for Retinal Image Non-Uniform Illumination Removal (Li et al.) [(paper)](https://arxiv.org/pdf/2008.02984v1.pdf) 
* 2020 - Pathological myopia classification with simultaneous lesion segmentation using deep learning (Hemelings et al.) [(paper)](https://arxiv.org/pdf/2006.02813v1.pdf) 
* 2020 - Progressive Adversarial Semantic Segmentation (Imran et al.) [(paper)](https://arxiv.org/pdf/2005.04311v1.pdf) 
* 2020 - Regression and Learning with Pixel-wise Attention for Retinal Fundus Glaucoma Segmentation and Detection (Liu et al.) [(paper)](https://arxiv.org/pdf/2001.01815v1.pdf) [(code)](https://github.com/archit31uniyal/DC-Gnet)
* 2020 - Residual Spatial Attention Network for Retinal Vessel Segmentation (Guo et al.) [(paper)](https://arxiv.org/pdf/2009.08829v1.pdf) [(code)](https://github.com/clguo/RSAN)
* 2020 - Rethinking the Extraction and Interaction of Multi-Scale Features for Vessel Segmentation (Wu et al.) [(paper)](https://arxiv.org/pdf/2010.04428v1.pdf) 
* 2020 - Retinal Image Segmentation with a Structure-Texture Demixing Network (Zhang et al.) [(paper)](https://arxiv.org/pdf/2008.00817v1.pdf) 
* 2020 - Retinal vessel segmentation by probing adaptive to lighting variations (Noyel et al.) [(paper)](https://arxiv.org/pdf/2004.13992v1.pdf) 
* 2020 - Robust Retinal Vessel Segmentation from a Data Augmentation Perspective (Sun et al.) [(paper)](https://arxiv.org/pdf/2007.15883v1.pdf) [(code)](https://github.com/PaddlePaddle/Research)
* 2020 - Robust Segmentation of Optic Disc and Cup from Fundus Images Using Deep Neural Networks (Manjunath et al.) [(paper)](https://arxiv.org/pdf/2012.07128v1.pdf) 
* 2020 - SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation (Guo et al.) [(paper)](https://arxiv.org/pdf/2004.03696v3.pdf) [(code)](https://github.com/clguo/SA-UNet)
* 2020 - SESV: Accurate Medical Image Segmentation byPredicting and Correcting Errors (Xia) [(paper)](https://ieeexplore.ieee.org/abstract/document/9201384) 
* 2020 - Supervised Segmentation of Retinal Vessel Structures Using ANN (Kaya et al.) [(paper)](https://arxiv.org/pdf/2001.05549v1.pdf) 
* 2020 - The Little W-Net That Could: State-of-the-Art Retinal Vessel Segmentation with Minimalistic Models (Galdran et al.) [(paper)](https://arxiv.org/pdf/2009.01907v1.pdf) [(code)](https://github.com/agaldran/lwnet)
* 2020 - The Unreasonable Effectiveness of Encoder-Decoder Networks for Retinal Vessel Segmentation (Browatzki et al.) [(paper)](https://arxiv.org/pdf/2011.12643v1.pdf) [(code)](https://github.com/browatbn2/VLight)
* 2020 - Transfer Learning Through Weighted Loss Function and Group Normalization for Vessel Segmentation from Retinal Images (Sarhan et al.) [(paper)](https://arxiv.org/pdf/2012.09250v1.pdf) [(code)](https://github.com/AbdullahSarhan/ICPRVessels)
* 2020 - U-Net with Graph Based Smoothing Regularizer for Small Vessel Segmentation on Fundus Image (Hakim et al.) [(paper)](https://arxiv.org/pdf/2009.07567v1.pdf) 
* 2020 - Unsupervised Learning of Local Discriminative Representation for Medical Images (Chen et al.) [(paper)](https://arxiv.org/pdf/2012.09333v2.pdf) [(code)](https://github.com/HuaiChen-1994/LDLearning)
* 2020 - Utilizing Transfer Learning and a Customized Loss Function for Optic Disc Segmentation from Retinal Images (Sarhan et al.) [(paper)](https://arxiv.org/pdf/2010.00583v1.pdf) [(code)](https://github.com/AbdullahSarhan/ACCVDiscSegmentation)
* 2020 - W-net: Simultaneous segmentation of multi-anatomical retinal structures using a multi-task deep neural network (Zhao et al.) [(paper)](https://arxiv.org/pdf/2006.06277v1.pdf) 
* 2019 - A Refined Equilibrium Generative Adversarial Network for Retinal Vessel Segmentation (Zhou et al.) [(paper)](https://arxiv.org/pdf/1909.11936v2.pdf) 
* 2019 - A Segmentation-Oriented Inter-Class Transfer Method: Application to Retinal Vessel Segmentation (Shi et al.) [(paper)](https://arxiv.org/pdf/1906.08501v1.pdf) 
* 2019 - A Two Stage GAN for High Resolution Retinal Image Generation and Segmentation (Andreini et al.) [(paper)](https://arxiv.org/pdf/1907.12296v1.pdf) 
* 2019 - A deep learning model for segmentation of geographic atrophy to study its long-term natural history (Liefers et al.) [(paper)](http://arxiv.org/pdf/1908.05621v1.pdf) 
* 2019 - Accurate Retinal Vessel Segmentation via Octave Convolution Neural Network (Fan et al.) [(paper)](https://arxiv.org/pdf/1906.12193v8.pdf) [(code)](https://github.com/JiajieMo/OctaveUNet)
* 2019 - Adversarial Learning with Multiscale Features and Kernel Factorization for Retinal Blood Vessel Segmentation (Akram et al.) [(paper)](https://arxiv.org/pdf/1907.02742v1.pdf) 
* 2019 - Attention Guided Network for Retinal Image Segmentation (Zhang et al.) [(paper)](https://arxiv.org/pdf/1907.12930v3.pdf) [(code)](https://github.com/HzFu/MNet_DeepCDR)
* 2019 - Automated Segmentation of the Optic Disk and Cup using Dual-Stage Fully Convolutional Networks (Bi et al.) [(paper)](http://arxiv.org/pdf/1902.04713v1.pdf) 
* 2019 - Automated retinal vessel segmentation based on morphological preprocessing and 2D-Gabor wavelets (Kumar et al.) [(paper)](https://arxiv.org/pdf/1908.04123v1.pdf) 
* 2019 - Blood Vessel Detection using Modified Multiscale MF-FDOG Filters for Diabetic Retinopathy (Mallick et al.) [(paper)](https://arxiv.org/pdf/1910.12028v1.pdf) 
* 2019 - CFEA: Collaborative Feature Ensembling Adaptation for Domain Adaptation in Unsupervised Optic Disc and Cup Segmentation (Liu et al.) [(paper)](https://arxiv.org/pdf/1910.07638v1.pdf) [(code)](https://github.com/PengyiZhang/MIADeepSSL)
* 2019 - Collaborative Learning of Semi-Supervised Segmentation and Classification for Medical Images (Zhou et al.) [(paper)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_Collaborative_Learning_of_Semi-Supervised_Segmentation_and_Classification_for_Medical_Images_CVPR_2019_paper.pdf) 
* 2019 - Connection Sensitive Attention U-NET for Accurate Retinal Vessel Segmentation (Li et al.) [(paper)](http://arxiv.org/pdf/1903.05558v2.pdf) 
* 2019 - Context-Aware Spatio-Recurrent Curvilinear Structure Segmentation (Wang et al.) [(paper)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Context-Aware_Spatio-Recurrent_Curvilinear_Structure_Segmentation_CVPR_2019_paper.pdf) 
* 2019 - DDNet: Cartesian-polar Dual-domain Network for the Joint Optic Disc and Cup Segmentation (Liu et al.) [(paper)](http://arxiv.org/pdf/1904.08773v1.pdf) 
* 2019 - DR-GAN: Conditional Generative Adversarial Network for Fine-Grained Lesion Synthesis on Diabetic Retinopathy Images (Zhou et al.) [(paper)](https://arxiv.org/pdf/1912.04670v3.pdf) 
* 2019 - Deep Retinal Image Segmentation with Regularization Under Geometric Priors (Cherukuri et al.) [(paper)](http://arxiv.org/pdf/1909.09175v1.pdf) 
* 2019 - Dense Dilated Network with Probability Regularized Walk for Vessel Detection (Mou et al.) [(paper)](https://arxiv.org/pdf/1910.12010v1.pdf) 
* 2019 - Dynamic Deep Networks for Retinal Vessel Segmentation (Khanal et al.) [(paper)](http://arxiv.org/pdf/1903.07803v2.pdf) [(code)](https://github.com/sraashis/ature)
* 2019 - ET-Net: A Generic Edge-aTtention Guidance Network for Medical Image Segmentation (Zhang et al.) [(paper)](https://arxiv.org/pdf/1907.10936v1.pdf) 
* 2019 - ErrorNet: Learning error representations from limited data to improve vascular segmentation (Tajbakhsh et al.) [(paper)](https://arxiv.org/pdf/1910.04814v4.pdf) 
* 2019 - From Patch to Image Segmentation using Fully Convolutional Networks -- Application to Retinal Images (Sekou et al.) [(paper)](https://arxiv.org/pdf/1904.03892v2.pdf) [(code)](https://github.com/Taib/patch2image)
* 2019 - Fully Convolutional Networks for Monocular Retinal Depth Estimation and Optic Disc-Cup Segmentation (Shankaranarayana et al.) [(paper)](http://arxiv.org/pdf/1902.01040v1.pdf) 
* 2019 - Fully Convolutional Neural Network for Semantic Segmentation of Anatomical Structure and Pathologies in Colour Fundus Images Associated with Diabetic Retinopathy (Saha et al.) [(paper)](http://arxiv.org/pdf/1902.03122v1.pdf) 
* 2019 - Generative Adversarial Networks And Domain Adaptation For Training Data Independent Image Registration (Mahapatra) [(paper)](https://arxiv.org/pdf/1910.08593v2.pdf) 
* 2019 - HybridNetSeg: A Compact Hybrid Network for Retinal Vessel Segmentation (Luo et al.) [(paper)](https://arxiv.org/pdf/1911.09982v1.pdf) 
* 2019 - IterNet: Retinal Image Segmentation Utilizing Structural Redundancy in Vessel Networks (Li et al.) [(paper)](https://arxiv.org/pdf/1912.05763v1.pdf) [(code)](https://github.com/conscienceli/IterNet)
* 2019 - Joint segmentation and classification of retinal arteries/veins from fundus images (Girard et al.) [(paper)](http://arxiv.org/pdf/1903.01330v1.pdf) 
* 2019 - Label Refinement with an Iterative Generative Adversarial Network for Boosting Retinal Vessel Segmentation (Yang et al.) [(paper)](http://arxiv.org/pdf/1912.02589v1.pdf) 
* 2019 - Learn to Segment Retinal Lesions and Beyond (Wei et al.) [(paper)](https://arxiv.org/pdf/1912.11619v3.pdf) [(code)](https://github.com/WeiQijie/retinal-lesions)
* 2019 - Learning Mutually Local-global U-nets For High-resolution Retinal Lesion Segmentation in Fundus Images (Yan et al.) [(paper)](http://arxiv.org/pdf/1901.06047v1.pdf) 
* 2019 - Multi-scale Microaneurysms Segmentation Using Embedding Triplet Loss (Sarhan et al.) [(paper)](https://arxiv.org/pdf/1904.12732v2.pdf) 
* 2019 - On the Evaluation and Real-World Usage Scenarios of Deep Vessel Segmentation for Retinography (Laibacher et al.) [(paper)](https://arxiv.org/pdf/1909.03856v3.pdf) [(code)](https://github.com/bioidiap/bob.ip.binseg)
* 2019 - Particle Swarm Optimization for Great Enhancement in Semi-Supervised Retinal Vessel Segmentation with Generative Adversarial Networks (Huo) [(paper)](https://arxiv.org/pdf/1906.07084v2.pdf) 
* 2019 - Patch-based Generative Adversarial Network Towards Retinal Vessel Segmentation (Abbas et al.) [(paper)](https://arxiv.org/pdf/1912.10377v1.pdf) 
* 2019 - Patch-based Output Space Adversarial Learning for Joint Optic Disc and Cup Segmentation (Wang et al.) [(paper)](http://arxiv.org/pdf/1902.07519v1.pdf) 
* 2019 - Progressive Generative Adversarial Networks for Medical Image Super resolution (Mahapatra et al.) [(paper)](http://arxiv.org/pdf/1902.02144v2.pdf) 
* 2019 - REFUGE Challenge: A Unified Framework for Evaluating Automated Methods for Glaucoma Assessment from Fundus Photographs (Orlando et al.) [(paper)](https://arxiv.org/pdf/1910.03667v1.pdf) 
* 2019 - Retinal Vessel Segmentation based on Fully Convolutional Networks (Liu) [(paper)](https://arxiv.org/pdf/1911.09915v1.pdf) [(code)](https://github.com/zhengyuan-liu/Retinal-Vessel-Segmentation)
* 2019 - Retinal Vessels Segmentation Based on Dilated Multi-Scale Convolutional Neural Network (Jiang et al.) [(paper)](http://arxiv.org/pdf/1904.05644v1.pdf) 
* 2019 - Segmentation of blood vessels in retinal fundus images (Straat et al.) [(paper)](https://arxiv.org/pdf/1905.12596v1.pdf) 
* 2019 - Transformation Consistent Self-ensembling Model for Semi-supervised Medical Image Segmentation (Li et al.) [(paper)](https://arxiv.org/pdf/1903.00348v3.pdf) 
* 2019 - What Do We Really Need? Degenerating U-Net on Retinal Vessel Segmentation (Fu et al.) [(paper)](https://arxiv.org/pdf/1911.02660v1.pdf) 
* 2018 - A Retinal Image Enhancement Technique for Blood Vessel Segmentation Algorithm (Bandara et al.) [(paper)](http://arxiv.org/pdf/1803.00036v1.pdf) 
* 2018 - Application of Deep Learning in Fundus Image Processing for Ophthalmic Diagnosis -- A Review (Sengupta et al.) [(paper)](https://arxiv.org/pdf/1812.07101v3.pdf) 
* 2018 - Auto-Classification of Retinal Diseases in the Limit of Sparse Data Using a Two-Streams Machine Learning Model (Yang et al.) [(paper)](http://arxiv.org/pdf/1808.05754v4.pdf) [(code)](https://github.com/huckiyang/EyeNet2)
* 2018 - Automated segmentaiton and classification of arterioles and venules using Cascading Dilated Convolutional Neural Networks (Li et al.) [(paper)](http://arxiv.org/pdf/1812.00137v1.pdf) 
* 2018 - BTS-DSN: Deeply Supervised Neural Network with Short Connections for Retinal Vessel Segmentation (Guo et al.) [(paper)](https://arxiv.org/pdf/1803.03963v2.pdf) [(code)](https://github.com/guomugong/BTS-DSN)
* 2018 - Brain-inspired robust delineation operator (Strisciuglio et al.) [(paper)](http://arxiv.org/pdf/1811.10240v1.pdf) [(code)](https://gitlab.com/nicstrisc/RUSTICO)
* 2018 - DUNet: A deformable network for retinal vessel segmentation (Jin et al.) [(paper)](http://arxiv.org/pdf/1811.01206v1.pdf) 
* 2018 - Deep Learning based Computer-Aided Diagnosis Systems for Diabetic Retinopathy: A Survey (Asiri et al.) [(paper)](https://arxiv.org/pdf/1811.01238v2.pdf) 
* 2018 - Deep Vessel Segmentation By Learning Graphical Connectivity (Shin et al.) [(paper)](http://arxiv.org/pdf/1806.02279v1.pdf) [(code)](https://github.com/syshin1014/VGN)
* 2018 - Deep supervision with additional labels for retinal vessel segmentation task (Zhang et al.) [(paper)](http://arxiv.org/pdf/1806.02132v3.pdf) 
* 2018 - Embedded deep learning in ophthalmology: Making ophthalmic imaging smarter (Teikari et al.) [(paper)](http://arxiv.org/pdf/1810.05874v2.pdf) 
* 2018 - GAN Based Medical Image Registration (Mahapatra) [(paper)](https://arxiv.org/pdf/1805.02369v4.pdf) 
* 2018 - High-resolution medical image synthesis using progressively grown generative adversarial networks (Beers et al.) [(paper)](http://arxiv.org/pdf/1805.03144v2.pdf) 
* 2018 - Iterative Deep Learning for Road Topology Extraction (Ventura et al.) [(paper)](http://arxiv.org/pdf/1808.09814v1.pdf) [(code)](https://github.com/carlesventura/iterative-deep-learning)
* 2018 - Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation (Fu et al.) [(paper)](http://arxiv.org/pdf/1801.00926v3.pdf) [(code)](https://github.com/HzFu/MNet_DeepCDR)
* 2018 - Low complexity convolutional neural network for vessel segmentation in portable retinal diagnostic devices (Hajabdollahi et al.) [(paper)](http://arxiv.org/pdf/1802.07804v1.pdf) 
* 2018 - M2U-Net: Effective and Efficient Retinal Vessel Segmentation for Resource-Constrained Environments (Laibacher et al.) [(paper)](http://arxiv.org/pdf/1811.07738v3.pdf) 
* 2018 - MedAL: Deep Active Learning Sampling Method for Medical Image Analysis (Smailagic et al.) [(paper)](http://arxiv.org/pdf/1809.09287v2.pdf) 
* 2018 - Minimal Paths for Tubular Structure Segmentation with Coherence Penalty and Adaptive Anisotropy (Chen et al.) [(paper)](http://arxiv.org/pdf/1809.07987v4.pdf) 
* 2018 - Multi-scale Neural Networks for Retinal Blood Vessels Segmentation (Zhang et al.) [(paper)](http://arxiv.org/pdf/1804.04206v1.pdf) 
* 2018 - RetinaMatch: Efficient Template Matching of Retina Images for Teleophthalmology (Gong et al.) [(paper)](http://arxiv.org/pdf/1811.11874v1.pdf) 
* 2018 - Retinal Optic Disc Segmentation using Conditional Generative Adversarial Network (Singh et al.) [(paper)](http://arxiv.org/pdf/1806.03905v1.pdf) 
* 2018 - Retinal Vessel Segmentation Based on Conditional Deep Convolutional Generative Adversarial Networks (Jiang et al.) [(paper)](http://arxiv.org/pdf/1805.04224v1.pdf) 
* 2018 - Retinal Vessel Segmentation under Extreme Low Annotation: A Generative Adversarial Network Approach (Lahiri et al.) [(paper)](http://arxiv.org/pdf/1809.01348v1.pdf) 
* 2018 - Retinal vessel segmentation based on Fully Convolutional Neural Networks (Oliveira et al.) [(paper)](http://arxiv.org/pdf/1812.07110v2.pdf) [(code)](https://github.com/americofmoliveira/VesselSegmentation_ESWA)
* 2018 - Scale Space Approximation in Convolutional Neural Networks for Retinal Vessel Segmentation (Noh et al.) [(paper)](http://arxiv.org/pdf/1806.09230v2.pdf) 
* 2018 - Structure-preserving Guided Retinal Image Filtering and Its Application for Optic Disc Analysis (Cheng et al.) [(paper)](http://arxiv.org/pdf/1805.06625v2.pdf) 
* 2018 - Towards a glaucoma risk index based on simulated hemodynamics from fundus images (Orlando et al.) [(paper)](http://arxiv.org/pdf/1805.10273v4.pdf) 
* 2018 - UOLO - automatic object detection and segmentation in biomedical images (Araújo et al.) [(paper)](http://arxiv.org/pdf/1810.05729v1.pdf) 
* 2017 - A Hierarchical Image Matting Model for Blood Vessel Segmentation in Fundus images (Fan et al.) [(paper)](http://arxiv.org/pdf/1701.00892v3.pdf) 
* 2017 - A Labeling-Free Approach to Supervising Deep Neural Networks for Retinal Blood Vessel Segmentation (Chen) [(paper)](http://arxiv.org/pdf/1704.07502v2.pdf) 
* 2017 - A Recursive Bayesian Approach To Describe Retinal Vasculature Geometry (Uslu et al.) [(paper)](http://arxiv.org/pdf/1711.10521v1.pdf) 
* 2017 - An adaptive thresholding approach for automatic optic disk segmentation (Ghadiri et al.) [(paper)](http://arxiv.org/pdf/1710.05104v1.pdf) 
* 2017 - Automatic Segmentation of Retinal Vasculature (Chalakkal et al.) [(paper)](http://arxiv.org/pdf/1707.06323v1.pdf) 
* 2017 - Delineation of line patterns in images using B-COSFIRE filters (Strisciuglio et al.) [(paper)](http://arxiv.org/pdf/1707.07438v1.pdf) 
* 2017 - Iterative Deep Learning for Network Topology Extraction (Ventura et al.) [(paper)](http://arxiv.org/pdf/1712.01217v1.pdf) 
* 2017 - Optic Disc and Cup Segmentation Methods for Glaucoma Detection with Modification of U-Net Convolutional Neural Network (Sevastopolsky) [(paper)](http://arxiv.org/pdf/1704.00979v1.pdf) [(code)](https://github.com/seva100/optic-nerve-cnn)
* 2017 - PixelBNN: Augmenting the PixelCNN with batch normalization and the presentation of a fast architecture for retinal vessel segmentation (Leopold et al.) [(paper)](http://arxiv.org/pdf/1712.06742v1.pdf) 
* 2017 - Retinal Vasculature Segmentation Using Local Saliency Maps and Generative Adversarial Networks For Image Super Resolution (Mahapatra et al.) [(paper)](http://arxiv.org/pdf/1710.04783v3.pdf) 
* 2017 - Retinal Vessel Segmentation in Fundoscopic Images with Generative Adversarial Networks (Son et al.) [(paper)](http://arxiv.org/pdf/1706.09318v1.pdf) [(code)](https://github.com/ChengBinJin/V-GAN-tensorflow)
* 2017 - Segmentation of optic disc, fovea and retinal vasculature using a single convolutional neural network (Tan et al.) [(paper)](http://arxiv.org/pdf/1702.00509v1.pdf) 
* 2017 - Seminar Innovation Management - Winter Term 2017 (Häusler et al.) [(paper)](http://arxiv.org/pdf/1708.09706v1.pdf) 
* 2017 - Synthesizing Filamentary Structured Images with GANs (Zhao et al.) [(paper)](http://arxiv.org/pdf/1706.02185v1.pdf) [(code)](https://github.com/ibrahimayaz/Fila-GAN)
* 2017 - The Multiscale Bowler-Hat Transform for Blood Vessel Enhancement in Retinal Images (Sazak et al.) [(paper)](http://arxiv.org/pdf/1709.05495v3.pdf) 
* 2017 - Towards Adversarial Retinal Image Synthesis (Costa et al.) [(paper)](http://arxiv.org/pdf/1701.08974v1.pdf) [(code)](https://github.com/costapt/vess2ret)
* 2016 - A Fully Convolutional Neural Network based Structured Prediction Approach Towards the Retinal Vessel Segmentation (Dasgupta et al.) [(paper)](http://arxiv.org/pdf/1611.02064v2.pdf) 
* 2016 - Consensus Based Medical Image Segmentation Using Semi-Supervised Learning And Graph Cuts (Mahapatra) [(paper)](http://arxiv.org/pdf/1612.02166v3.pdf) 
* 2016 - Curvature Integration in a 5D Kernel for Extracting Vessel Connections in Retinal Images (Abbasi-Sureshjani et al.) [(paper)](http://arxiv.org/pdf/1608.08049v3.pdf) 
* 2016 - Deep Neural Ensemble for Retinal Vessel Segmentation in Fundus Images towards Achieving Label-free Angiography (Lahiri et al.) [(paper)](http://arxiv.org/pdf/1609.05871v1.pdf) 
* 2016 - Deep Retinal Image Understanding (Maninis et al.) [(paper)](http://arxiv.org/pdf/1609.01103v1.pdf) [(code)](https://github.com/PB17151764/2020UM-Summer-Research)
* 2016 - Memory Efficient Multi-Scale Line Detector Architecture for Retinal Blood Vessel Segmentation (Bendaoudi et al.) [(paper)](http://arxiv.org/pdf/1612.09524v1.pdf) 
* 2016 - Retinal Vessel Segmentation Using A New Topological Method (Brooks) [(paper)](http://arxiv.org/pdf/1608.01339v1.pdf) 
* 2016 - Retrieving challenging vessel connections in retinal images by line co-occurrence statistics (Abbasi-Sureshjani et al.) [(paper)](http://arxiv.org/pdf/1610.06368v1.pdf) 
* 2015 - Analysis of Vessel Connectivities in Retinal Images by Cortically Inspired Spectral Clustering (Favali et al.) [(paper)](http://arxiv.org/pdf/1512.06559v2.pdf) 
* 2015 - Locally Adaptive Frames in the Roto-Translation Group and their Applications in Medical Imaging (Duits et al.) [(paper)](http://arxiv.org/pdf/1502.08002v7.pdf) 
* 2014 - Ant Colony based Feature Selection Heuristics for Retinal Vessel Segmentation (Asad et al.) [(paper)](http://arxiv.org/pdf/1403.1735v1.pdf) 
* 2014 - Drishti-GS: Retinal image dataset for optic nerve head(ONH) segmentation (Sivaswamy et al.) [(paper)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6867807&casa_token=30jYyY-ICYsAAAAA:ma87uhjGikfrdBZQw1r0-o_ck-KGYW4yTw3MVV4XvFtrYoEo8z9UHuzzvVPeEt-k342biqY&tag=1) 
* 2014 - GIMP and Wavelets for Medical Image Processing: Enhancing Images of the Fundus of the Eye (Sparavigna) [(paper)](http://arxiv.org/pdf/1408.4703v1.pdf) 
* 2014 - Vesselness via Multiple Scale Orientation Scores (Hannink et al.) [(paper)](http://arxiv.org/pdf/1402.4963v4.pdf) 
* 2013 - A Novel Retinal Vessel Segmentation Based On Histogram Transformation Using 2-D Morlet Wavelet and Supervised Classification (Fazli et al.) [(paper)](http://arxiv.org/pdf/1312.7557v1.pdf) 
* 2013 - BW - Eye Ophthalmologic decision support system based on clinical workflow and data mining techniques-image registration algorithm (Martins) [(paper)](http://arxiv.org/pdf/1312.4752v1.pdf) 
* 2012 - FCM Based Blood Vessel Segmentation Method for Retinal Images (Dey et al.) [(paper)](http://arxiv.org/pdf/1209.1181v1.pdf) 
* 2005 - Retinal Vessel Segmentation Using the 2-D Morlet Wavelet and Supervised Classification (Soares et al.) [(paper)](http://arxiv.org/pdf/cs/0510001v2.pdf) 
<br/><br/>

---
<a name="retinal-oct-classification"></a>
### Retinal OCT Classification
---
* 2021 - Euler Characteristic Surfaces (Beltramo et al.) [(paper)](http://arxiv.org/pdf/2102.08260v1.pdf) 
* 2021 - I-ODA, Real-World Multi-modal Longitudinal Data for OphthalmicApplications (Mojab et al.) [(paper)](https://arxiv.org/pdf/2104.02609v1.pdf) 
* 2021 - Medical Image Quality Metrics for Foveated Model Observers (Lago et al.) [(paper)](https://arxiv.org/pdf/2102.05178v1.pdf) 
* 2020 - A scoping review of transfer learning research on medical image analysis using ImageNet (Morid et al.) [(paper)](https://arxiv.org/pdf/2004.13175v5.pdf) 
* 2020 - AGE Challenge: Angle Closure Glaucoma Evaluation in Anterior Segment Optical Coherence Tomography (Fu et al.) [(paper)](https://arxiv.org/pdf/2005.02258v3.pdf) 
* 2020 - Analysis of Hand-Crafted and Automatic-Learned Features for Glaucoma Detection Through Raw Circmpapillary OCT Images (García et al.) [(paper)](https://arxiv.org/pdf/2009.04190v1.pdf) 
* 2020 - Autonomously Navigating a Surgical Tool Inside the Eye by Learning from Demonstration (Kim et al.) [(paper)](https://arxiv.org/pdf/2011.07785v1.pdf) 
* 2020 - Conditional GAN for Prediction of Glaucoma Progression with Macular Optical Coherence Tomography (Hassan et al.) [(paper)](https://arxiv.org/pdf/2010.04552v1.pdf) 
* 2020 - DcardNet: Diabetic Retinopathy Classification at Multiple Levels Based on Structural and Angiographic Optical Coherence Tomography (Zang et al.) [(paper)](https://arxiv.org/pdf/2006.05480v2.pdf) 
* 2020 - Deep OCT Angiography Image Generation for Motion Artifact Suppression (Hossbach et al.) [(paper)](https://arxiv.org/pdf/2001.02512v1.pdf) 
* 2020 - Deep Sequential Feature Learning in Clinical Image Classification of Infectious Keratitis (Xu et al.) [(paper)](https://arxiv.org/pdf/2006.02666v1.pdf) 
* 2020 - Deep learning achieves perfect anomaly detection on 108,308 retinal images including unlearned diseases (Suzuki et al.) [(paper)](https://arxiv.org/pdf/2001.05859v5.pdf) [(code)](https://github.com/SAyaka0122/Deep-learning-based-binary-classifier)
* 2020 - Encoding Structure-Texture Relation with P-Net for Anomaly Detection in Retinal Images (Zhou et al.) [(paper)](https://arxiv.org/pdf/2008.03632v1.pdf) [(code)](https://github.com/ClancyZhou/P_Net_Anomaly_Detection)
* 2020 - ExplAIn: Explanatory Artificial Intelligence for Diabetic Retinopathy Diagnosis (Quellec et al.) [(paper)](https://arxiv.org/pdf/2008.05731v2.pdf) 
* 2020 - Focal Loss Analysis of Nerve Fiber Layer Reflectance for Glaucoma Diagnosis (Tan et al.) [(paper)](http://arxiv.org/pdf/2006.13522v1.pdf) 
* 2020 - Glaucoma Detection From Raw Circumapillary OCT Images Using Fully Convolutional Neural Networks (García et al.) [(paper)](https://arxiv.org/pdf/2006.00027v1.pdf) 
* 2020 - High-resolution wide-field OCT angiography with a self-navigation method to correct microsaccades and blinks (Wei et al.) [(paper)](http://arxiv.org/pdf/2004.04823v3.pdf) 
* 2020 - Improving Robustness using Joint Attention Network For Detecting Retinal Degeneration From Optical Coherence Tomography Images (Kamran et al.) [(paper)](https://arxiv.org/pdf/2005.08094v2.pdf) [(code)](https://github.com/SharifAmit/Robust_Joint_Attention)
* 2020 - Label-free, non-contact, in-vivo ophthalmic imaging using photoacoustic remote sensing microscopy (Hosseinaee et al.) [(paper)](http://arxiv.org/pdf/2009.06088v1.pdf) 
* 2020 - Matching the Clinical Reality: Accurate OCT-Based Diagnosis From Few Labels (Melnychuk et al.) [(paper)](https://arxiv.org/pdf/2010.12316v1.pdf) [(code)](https://github.com/Valentyn1997/oct-diagn-semi-supervised)
* 2020 - Multiresolution Knowledge Distillation for Anomaly Detection (Salehi et al.) [(paper)](https://arxiv.org/pdf/2011.11108v1.pdf) [(code)](https://github.com/Niousha12/Knowledge_Distillation_AD)
* 2020 - Optical Incoherence Tomography: a method to generate tomographic retinal cross-sections with non-interferometric imaging systems (Mecê et al.) [(paper)](http://arxiv.org/pdf/2002.12812v1.pdf) 
* 2020 - Quantitative and Qualitative Evaluation of Explainable Deep Learning Methods for Ophthalmic Diagnosis (Singh et al.) [(paper)](https://arxiv.org/pdf/2009.12648v2.pdf) 
* 2020 - Scientific Discovery by Generating Counterfactuals using Image Translation (Narayanaswamy et al.) [(paper)](https://arxiv.org/pdf/2007.05500v2.pdf) 
* 2020 - Shot-noise limited, supercontinuum based optical coherence tomography (S. et al.) [(paper)](http://arxiv.org/pdf/2010.05226v1.pdf) 
* 2020 - Towards Autonomous Eye Surgery by Combining Deep Imitation Learning with Optimal Control (Kim et al.) [(paper)](http://arxiv.org/pdf/2011.07778v1.pdf) 
* 2020 - Video Coding for Machines: A Paradigm of Collaborative Compression and Intelligent Analytics (Duan et al.) [(paper)](http://arxiv.org/pdf/2001.03569v2.pdf) 
* 2019 - Angle-Closure Detection in Anterior Segment OCT based on Multi-Level Deep Network (Fu et al.) [(paper)](http://arxiv.org/pdf/1902.03585v1.pdf) 
* 2019 - Assessment of Generative Adversarial Networks Model for Synthetic Optical Coherence Tomography Images of Retinal Disorders (Zheng et al.) [(paper)](http://arxiv.org/pdf/1910.09748v1.pdf) 
* 2019 - Classification of dry age-related macular degeneration and diabetic macular edema from optical coherence tomography images using dictionary learning (Mousavi et al.) [(paper)](http://arxiv.org/pdf/1903.06909v1.pdf) 
* 2019 - Comparisonal study of Deep Learning approaches on Retinal OCT Image (Tasnim et al.) [(paper)](https://arxiv.org/pdf/1912.07783v1.pdf) 
* 2019 - Disease classification of macular Optical Coherence Tomography scans using deep learning software: validation on independent, multi-centre data (Bhatia et al.) [(paper)](https://arxiv.org/pdf/1907.05164v1.pdf) 
* 2019 - Evaluation of Transfer Learning for Classification of: (1) Diabetic Retinopathy by Digital Fundus Photography and (2) Diabetic Macular Edema, Choroidal Neovascularization and Drusen by Optical Coherence Tomography (Gelman) [(paper)](http://arxiv.org/pdf/1902.04151v1.pdf) 
* 2019 - Finding New Diagnostic Information for Detecting Glaucoma using Neural Networks (Noury et al.) [(paper)](https://arxiv.org/pdf/1910.06302v2.pdf) 
* 2019 - Fused Detection of Retinal Biomarkers in OCT Volumes (Kurmann et al.) [(paper)](https://arxiv.org/pdf/1907.06955v1.pdf) 
* 2019 - Informing Computer Vision with Optical Illusions (Nematzadeh et al.) [(paper)](http://arxiv.org/pdf/1902.02922v1.pdf) 
* 2019 - Modeling Disease Progression In Retinal OCTs With Longitudinal Self-Supervised Learning (Rivail et al.) [(paper)](https://arxiv.org/pdf/1910.09420v3.pdf) 
* 2019 - Semantic denoising autoencoders for retinal optical coherence tomography (Laves et al.) [(paper)](http://arxiv.org/pdf/1903.09809v1.pdf) 
* 2019 - Sparse-GAN: Sparsity-constrained Generative Adversarial Network for Anomaly Detection in Retinal OCT Image (Zhou et al.) [(paper)](https://arxiv.org/pdf/1911.12527v3.pdf) 
* 2019 - Supervised machine learning based multi-task artificial intelligence classification of retinopathies (Alam et al.) [(paper)](http://arxiv.org/pdf/1905.04224v1.pdf) 
* 2019 - Transfer Learning for Automated OCTA Detection of Diabetic Retinopathy (Le et al.) [(paper)](http://arxiv.org/pdf/1910.01796v1.pdf) 
* 2019 - Transscleral Optical Phase Imaging of the Human Retina - TOPI (Laforest et al.) [(paper)](http://arxiv.org/pdf/1905.06877v1.pdf) 
* 2019 - Unifying Structure Analysis and Surrogate-driven Function Regression for Glaucoma OCT Image Screening (Wang et al.) [(paper)](https://arxiv.org/pdf/1907.12927v1.pdf) 
* 2018 - A feature agnostic approach for glaucoma detection in OCT volumes (Maetschke et al.) [(paper)](https://arxiv.org/pdf/1807.04855v4.pdf) [(code)](https://github.com/yuliytsank/glaucoma-project)
* 2018 - Deep learning architecture LightOCT for diagnostic decision support using optical coherence tomography images of biological samples (Butola et al.) [(paper)](http://arxiv.org/pdf/1812.02487v2.pdf) 
* 2018 - Fast 5DOF Needle Tracking in iOCT (Weiss et al.) [(paper)](http://arxiv.org/pdf/1802.06446v1.pdf) 
* 2018 - From Machine to Machine: An OCT-trained Deep Learning Algorithm for Objective Quantification of Glaucomatous Damage in Fundus Photographs (Medeiros et al.) [(paper)](http://arxiv.org/pdf/1810.10343v1.pdf) 
* 2018 - Towards Ophthalmologist Level Accurate Deep Learning System for OCT Screening and Diagnosis (Mrinal) [(paper)](http://arxiv.org/pdf/1812.07105v1.pdf) 
* 2018 - Towards Ophthalmologist Level Accurate Deep Learning System for OCT Screening and Diagnosis (Haloi) [(paper)](http://arxiv.org/pdf/1812.07105v1.pdf) 
* 2018 - Towards Robotic Eye Surgery: Marker-free, Online Hand-eye Calibration using Optical Coherence Tomography Images (Zhou et al.) [(paper)](http://arxiv.org/pdf/1808.05805v1.pdf) 
* 2017 - Learning Asymmetric and Local Features in Multi-Dimensional Data through Wavelets with Recursive Partitioning (Li et al.) [(paper)](https://arxiv.org/pdf/1711.00789v5.pdf) [(code)](https://github.com/MaStatLab/WARP)
* 2017 - Quantum Biometrics with Retinal Photon Counting (Loulakis et al.) [(paper)](http://arxiv.org/pdf/1704.04367v2.pdf) 
* 2017 - Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (Schlegl et al.) [(paper)](http://arxiv.org/pdf/1703.05921v1.pdf) [(code)](https://github.com/xtarx/Unsupervised-Anomaly-Detection-with-Generative-Adversarial-Networks)
* 2016 - Deep learning is effective for the classification of OCT images of normal versus Age-related Macular Degeneration (Lee et al.) [(paper)](http://arxiv.org/pdf/1612.04891v1.pdf) 
* 2016 - RetiNet: Automatic AMD identification in OCT volumetric data (Apostolopoulos et al.) [(paper)](http://arxiv.org/pdf/1610.03628v1.pdf) 
* 2014 - Revealing cell assemblies at multiple levels of granularity (Billeh et al.) [(paper)](http://arxiv.org/pdf/1411.2103v1.pdf) 
<br/><br/>

---
<a name="retinal-oct-segmentation"></a>
### Retinal OCT Segmentation
---
* 2021 - 3D Vessel Reconstruction in OCT-Angiography via Depth Map Estimation (Yu et al.) [(paper)](https://arxiv.org/pdf/2102.13588v1.pdf) 
* 2021 - Automated segmentation of choroidal layers from 3-dimensional macular optical coherence tomography scans (Lee et al.) [(paper)](https://arxiv.org/pdf/2103.06425v1.pdf) 
* 2021 - Every Annotation Counts: Multi-label Deep Supervision for Medical Image Segmentation (Reiß et al.) [(paper)](https://arxiv.org/pdf/2104.13243v1.pdf) 
* 2021 - Fractal Dimension and Retinal Pathology: A Meta-analysis (Yu et al.) [(paper)](https://arxiv.org/pdf/2101.08815v1.pdf) 
* 2021 - Multi-scale GCN-assisted two-stage network for joint segmentation of retinal layers and disc in peripapillary OCT images (Li et al.) [(paper)](https://arxiv.org/pdf/2102.04799v1.pdf) [(code)](https://github.com/Jiaxuan-Li/MGU-Net)
* 2021 - Pointwise visual field estimation from optical coherence tomography in glaucoma: a structure-function analysis using deep learning (Hemelings et al.) [(paper)](https://arxiv.org/pdf/2106.03793v1.pdf) 
* 2021 - SVS-net: A Novel Semantic Segmentation Network in Optical Coherence Tomography Angiography Images (Lee et al.) [(paper)](https://arxiv.org/pdf/2104.07083v3.pdf) 
* 2021 - Uncertainty guided semi-supervised segmentation of retinal layers in OCT images (Sedai et al.) [(paper)](https://arxiv.org/pdf/2103.02083v1.pdf) 
* 2020 - Assignment Flow for Order-Constrained OCT Segmentation (Sitenko et al.) [(paper)](https://arxiv.org/pdf/2009.04632v1.pdf) 
* 2020 - Automated segmentation of retinal fluid volumes from structural and angiographic optical coherence tomography using deep learning (Guo et al.) [(paper)](https://arxiv.org/pdf/2006.02569v1.pdf) 
* 2020 - Automatic Segmentation and Visualization of Choroid in OCT with Knowledge Infused Deep Learning (Zhang et al.) [(paper)](http://arxiv.org/pdf/2002.04712v2.pdf) 
* 2020 - Clinically Verified Hybrid Deep Learning System for Retinal Ganglion Cells Aware Grading of Glaucomatous Progression (Raja et al.) [(paper)](https://arxiv.org/pdf/2010.03872v1.pdf) [(code)](https://github.com/taimurhassan/rag-net-v2)
* 2020 - Deep iterative vessel segmentation in OCT angiography (Pissas et al.) [(paper)](https://www.osapublishing.org/boe/viewmedia.cfm?uri=boe-11-5-2490&seq=0) [(code)](https://github.com/RViMLab/BOE2020-OCTA-vessel-segmentation)
* 2020 - Describing the Structural Phenotype of the Glaucomatous Optic Nerve Head Using Artificial Intelligence (Panda et al.) [(paper)](https://arxiv.org/pdf/2012.09755v1.pdf) 
* 2020 - Dictionary-based Method for Vascular Segmentation for OCTA Images (Engberg et al.) [(paper)](https://arxiv.org/pdf/2002.03945v1.pdf) 
* 2020 - Efficient OCT Image Segmentation Using Neural Architecture Search (Gheshlaghi et al.) [(paper)](https://arxiv.org/pdf/2007.14790v1.pdf) 
* 2020 - Efficient and high accuracy 3-D OCT angiography motion correction in pathology (Ploner et al.) [(paper)](https://arxiv.org/pdf/2010.06931v1.pdf) 
* 2020 - Exploiting the Transferability of Deep Learning Systems Across Multi-modal Retinal Scans for Extracting Retinopathy Lesions (Hassan et al.) [(paper)](https://arxiv.org/pdf/2006.02662v2.pdf) 
* 2020 - Fast 3-dimensional estimation of the Foveal Avascular Zone from OCTA (Ometto et al.) [(paper)](https://arxiv.org/pdf/2012.09945v1.pdf) 
* 2020 - Few Shot Learning Framework to Reduce Inter-observer Variability in Medical Images (Roychowdhury) [(paper)](https://arxiv.org/pdf/2008.02952v1.pdf) [(code)](https://github.com/sohiniroych/Paralllel-ESN-For-Image-Quality)
* 2020 - Globally Optimal Segmentation of Mutually Interacting Surfaces using Deep Learning (Xie et al.) [(paper)](https://arxiv.org/pdf/2007.01259v3.pdf) [(code)](https://github.com/Hui-Xie/DeepLearningSeg)
* 2020 - IPN-V2 and OCTA-500: Methodology and Dataset for Retinal Image Segmentation (Li et al.) [(paper)](https://arxiv.org/pdf/2012.07261v1.pdf) [(code)](https://github.com/chaosallen/IPNV2_pytorch)
* 2020 - In vivo imaging of human cornea with high-speed and high-resolution Fourier-domain full-field optical coherence tomography (Auksorius et al.) [(paper)](http://arxiv.org/pdf/2003.12085v1.pdf) 
* 2020 - Livelayer: A Semi-Automatic Software Program for Segmentation of Layers and Diabetic Macular Edema in Optical Coherence Tomography Images (Montazerin et al.) [(paper)](https://arxiv.org/pdf/2003.05916v3.pdf) 
* 2020 - Low Tensor Train- and Low Multilinear Rank Approximations for De-speckling and Compression of 3D Optical Coherence Tomography Images (Kopriva et al.) [(paper)](http://arxiv.org/pdf/2008.11414v2.pdf) 
* 2020 - MPG-Net: Multi-Prediction Guided Network for Segmentation of Retinal Layers in OCT Images (Fu et al.) [(paper)](https://arxiv.org/pdf/2009.13634v1.pdf) 
* 2020 - Microvasculature Segmentation and Inter-capillary Area Quantification of the Deep Vascular Complex using Transfer Learning (Lo et al.) [(paper)](https://arxiv.org/pdf/2003.09033v1.pdf) 
* 2020 - OCT-GAN: Single Step Shadow and Noise Removal from Optical Coherence Tomography Images of the Human Optic Nerve Head (Cheong et al.) [(paper)](https://arxiv.org/pdf/2010.11698v1.pdf) 
* 2020 - Pathological Retinal Region Segmentation From OCT Images Using Geometric Relation Based Augmentation (Mahapatra et al.) [(paper)](https://arxiv.org/pdf/2003.14119v3.pdf) 
* 2020 - Pseudo-real-time retinal layer segmentation for high-resolution adaptive optics optical coherence tomography (Janpongsri et al.) [(paper)](http://arxiv.org/pdf/2004.05264v1.pdf) 
* 2020 - ROSE: A Retinal OCT-Angiography Vessel Segmentation Dataset and New Model (Ma et al.) [(paper)](https://arxiv.org/pdf/2007.05201v2.pdf) [(code)](https://github.com/iMED-Lab/OCTA-Net-OCTA-Vessel-Segmentation-Network)
* 2020 - Recent Developments in Detection of Central Serous Retinopathy through Imaging and Artificial Intelligence Techniques A Review (Hassan et al.) [(paper)](https://arxiv.org/pdf/2012.10961v3.pdf) 
* 2020 - Segmentation of Retinal Low-Cost Optical Coherence Tomography Images using Deep Learning (Kepp et al.) [(paper)](https://arxiv.org/pdf/2001.08480v1.pdf) 
* 2020 - Self domain adapted network (He et al.) [(paper)](https://arxiv.org/pdf/2007.03162v1.pdf) [(code)](https://github.com/YufanHe/self-domain-adapted-network)
* 2020 - Superpixel-Guided Label Softening for Medical Image Segmentation (Li et al.) [(paper)](https://arxiv.org/pdf/2007.08897v1.pdf) 
* 2020 - Towards Label-Free 3D Segmentation of Optical Coherence Tomography Images of the Optic Nerve Head Using Deep Learning (Devalla et al.) [(paper)](https://arxiv.org/pdf/2002.09635v1.pdf) 
* 2019 - Accurate Tissue Interface Segmentation via Adversarial Pre-Segmentation of Anterior Segment OCT Images (Ouyang et al.) [(paper)](https://arxiv.org/pdf/1905.02378v1.pdf) 
* 2019 - An amplified-target loss approach for photoreceptor layer segmentation in pathological OCT scans (Orlando et al.) [(paper)](https://arxiv.org/pdf/1908.00764v2.pdf) 
* 2019 - Automated Segmentation of Optical Coherence Tomography Angiography Images: Benchmark Data and Clinically Relevant Metrics (Giarratano et al.) [(paper)](https://arxiv.org/pdf/1912.09978v2.pdf) [(code)](https://github.com/giaylenia/OCTA_segm_study)
* 2019 - BioNet: Infusing Biomarker Prior into Global-to-Local Network for Choroid Segmentation in Optical Coherence Tomography Images (Zhang et al.) [(paper)](https://arxiv.org/pdf/1912.05090v1.pdf) 
* 2019 - CE-Net: Context Encoder Network for 2D Medical Image Segmentation (Gu et al.) [(paper)](http://arxiv.org/pdf/1903.02740v1.pdf) [(code)](https://github.com/HzFu/MNet_DeepCDR)
* 2019 - Cascaded Deep Neural Networks for Retinal Layer Segmentation of Optical Coherence Tomography with Fluid Presence (Lu et al.) [(paper)](https://arxiv.org/pdf/1912.03418v1.pdf) 
* 2019 - Deep learning vessel segmentation and quantification of the foveal avascular zone using commercial and prototype OCT-A platforms (Heisler et al.) [(paper)](https://arxiv.org/pdf/1909.11289v1.pdf) 
* 2019 - DeshadowGAN: A Deep Learning Approach to Remove Shadows from Optical Coherence Tomography Images (Cheong et al.) [(paper)](https://arxiv.org/pdf/1910.02844v1.pdf) 
* 2019 - Exploiting Epistemic Uncertainty of Anatomy Segmentation for Anomaly Detection in Retinal OCT (Seeböck et al.) [(paper)](https://arxiv.org/pdf/1905.12806v1.pdf) 
* 2019 - Fluid segmentation in Neutrosophic domain (Rashno et al.) [(paper)](https://arxiv.org/pdf/1912.11540v1.pdf) 
* 2019 - High signal-to-noise ratio reconstruction of low bit-depth optical coherence tomography using deep learning (Hao et al.) [(paper)](http://arxiv.org/pdf/1910.05498v2.pdf) 
* 2019 - Inference of visual field test performance from OCT volumes using deep learning (Maetschke et al.) [(paper)](https://arxiv.org/pdf/1908.01428v3.pdf) 
* 2019 - Knowledge infused cascade convolutional neural network for segmenting retinal vessels in volumetric optical coherence tomography (Fang et al.) [(paper)](http://arxiv.org/pdf/1910.09187v1.pdf) 
* 2019 - Lesson Learnt: Modularization of Deep Networks Allow Cross-Modality Reuse (Fu et al.) [(paper)](https://arxiv.org/pdf/1911.02080v1.pdf) 
* 2019 - Multiclass segmentation as multitask learning for drusen segmentation in retinal optical coherence tomography (Asgari et al.) [(paper)](https://arxiv.org/pdf/1906.07679v2.pdf) 
* 2019 - Optic-Net: A Novel Convolutional Neural Network for Diagnosis of Retinal Diseases from Optical Tomography Images (Kamran et al.) [(paper)](https://arxiv.org/pdf/1910.05672v1.pdf) [(code)](https://github.com/SharifAmit/OpticNet-71)
* 2019 - The Channel Attention based Context Encoder Network for Inner Limiting Membrane Detection (Qiu et al.) [(paper)](https://arxiv.org/pdf/1908.04413v1.pdf) 
* 2019 - U-Net with spatial pyramid pooling for drusen segmentation in optical coherence tomography (Asgari et al.) [(paper)](https://arxiv.org/pdf/1912.05404v1.pdf) 
* 2019 - U2-Net: A Bayesian U-Net model with epistemic uncertainty feedback for photoreceptor layer segmentation in pathological OCT scans (Orlando et al.) [(paper)](https://arxiv.org/pdf/1901.07929v2.pdf) 
* 2019 - Uncertainty-Guided Domain Alignment for Layer Segmentation in OCT Images (Wang et al.) [(paper)](https://arxiv.org/pdf/1908.08242v2.pdf) 
* 2019 - Using CycleGANs for effectively reducing image variability across OCT devices and improving retinal fluid segmentation (Seeböck et al.) [(paper)](http://arxiv.org/pdf/1901.08379v2.pdf) 
* 2018 - A Comparison of Handcrafted and Deep Neural Network Feature Extraction for Classifying Optical Coherence Tomography (OCT) Images (Nugroho) [(paper)](http://arxiv.org/pdf/1809.03306v1.pdf) 
* 2018 - A Deep Learning Approach to Denoise Optical Coherence Tomography Images of the Optic Nerve Head (Devalla et al.) [(paper)](http://arxiv.org/pdf/1809.10589v1.pdf) 
* 2018 - A deep learning framework for segmentation of retinal layers from OCT images (Gopinath et al.) [(paper)](http://arxiv.org/pdf/1806.08859v1.pdf) 
* 2018 - Automatic Segmentation of Choroid Layer in EDI OCT Images Using Graph Theory in Neutrosophic Space (Salafian et al.) [(paper)](http://arxiv.org/pdf/1812.01989v1.pdf) 
* 2018 - Automatic segmentation of the Foveal Avascular Zone in ophthalmological OCT-A images (Díaz et al.) [(paper)](http://arxiv.org/pdf/1811.10374v1.pdf) 
* 2018 - DRUNET: A Dilated-Residual U-Net Deep Learning Network to Digitally Stain Optic Nerve Head Tissues in Optical Coherence Tomography Images (Devalla et al.) [(paper)](http://arxiv.org/pdf/1803.00232v1.pdf) 
* 2018 - Deep Learning based Retinal OCT Segmentation (Pekala et al.) [(paper)](http://arxiv.org/pdf/1801.09749v1.pdf) 
* 2018 - Fully Automated Segmentation of Hyperreflective Foci in Optical Coherence Tomography Images (Schlegl et al.) [(paper)](http://arxiv.org/pdf/1805.03278v1.pdf) 
* 2018 - Functional imaging of ganglion and receptor cells in living human retina by osmotic contrast (Pfäffle et al.) [(paper)](http://arxiv.org/pdf/1809.02812v1.pdf) 
* 2018 - Generating retinal flow maps from structural optical coherence tomography with artificial intelligence (Lee et al.) [(paper)](http://arxiv.org/pdf/1802.08925v1.pdf) 
* 2018 - Joint Segmentation and Uncertainty Visualization of Retinal Layers in Optical Coherence Tomography Images using Bayesian Deep Learning (Sedai et al.) [(paper)](http://arxiv.org/pdf/1809.04282v1.pdf) [(code)](https://github.com/ssedai026/uncertainty-segmentation)
* 2018 - Multi-Context Deep Network for Angle-Closure Glaucoma Screening in Anterior Segment OCT (Fu et al.) [(paper)](http://arxiv.org/pdf/1809.03239v1.pdf) 
* 2018 - Non-rigid image registration using spatially region-weighted correlation ratio and GPU-acceleration (Gong et al.) [(paper)](http://arxiv.org/pdf/1804.05061v1.pdf) 
* 2018 - OCT segmentation: Integrating open parametric contour model of the retinal layers and shape constraint to the Mumford-Shah functional (Duan et al.) [(paper)](http://arxiv.org/pdf/1808.02917v1.pdf) 
* 2018 - OCTID: Optical Coherence Tomography Image Database (Gholami et al.) [(paper)](https://arxiv.org/pdf/1812.07056v2.pdf) 
* 2018 - Three-dimensional Optical Coherence Tomography Image Denoising through Multi-input Fully-Convolutional Networks (Abbasi et al.) [(paper)](http://arxiv.org/pdf/1811.09022v2.pdf) 
* 2018 - Topology guaranteed segmentation of the human retina from OCT using convolutional neural networks (He et al.) [(paper)](http://arxiv.org/pdf/1803.05120v1.pdf) 
* 2017 - A Deep Learning Approach to Digitally Stain Optical Coherence Tomography Images of the Optic Nerve Head (Devalla et al.) [(paper)](http://arxiv.org/pdf/1707.07609v1.pdf) 
* 2017 - A Generalized Motion Pattern and FCN based approach for retinal fluid detection and segmentation (Yadav et al.) [(paper)](http://arxiv.org/pdf/1712.01073v1.pdf) 
* 2017 - Cystoid macular edema segmentation of Optical Coherence Tomography images using fully convolutional neural networks and fully connected CRFs (Bai et al.) [(paper)](http://arxiv.org/pdf/1709.05324v1.pdf) 
* 2017 - Pathological OCT Retinal Layer Segmentation using Branch Residual U-shape Networks (Apostolopoulos et al.) [(paper)](http://arxiv.org/pdf/1707.04931v1.pdf) 
* 2017 - ReLayNet: Retinal Layer and Fluid Segmentation of Macular Optical Coherence Tomography using Fully Convolutional Network (Roy et al.) [(paper)](http://arxiv.org/pdf/1704.02161v2.pdf) [(code)](https://github.com/Nikolay1998/relaynet_pytorch)
* 2017 - Retinal Fluid Segmentation and Detection in Optical Coherence Tomography Images using Fully Convolutional Neural Network (Lu et al.) [(paper)](http://arxiv.org/pdf/1710.04778v1.pdf) 
* 2017 - Segmentation of retinal cysts from Optical Coherence Tomography volumes via selective enhancement (Gopinath et al.) [(paper)](http://arxiv.org/pdf/1708.06197v2.pdf) 
* 2017 - Simultaneous Detection and Quantification of Retinal Fluid with Deep Learning (Morley et al.) [(paper)](http://arxiv.org/pdf/1708.05464v1.pdf) 
* 2016 - Automated OCT Segmentation for Images with DME (Roychowdhury et al.) [(paper)](http://arxiv.org/pdf/1610.07560v1.pdf) 
* 2016 - Automated Segmentation of Retinal Layers from Optical Coherent Tomography Images Using Geodesic Distance (Duan et al.) [(paper)](http://arxiv.org/pdf/1609.02214v1.pdf) 
* 2016 - Domain knowledge assisted cyst segmentation in OCT retinal images (Gopinath et al.) [(paper)](http://arxiv.org/pdf/1612.02675v1.pdf) 
* 2016 - Identifying and Categorizing Anomalies in Retinal Imaging Data (Seeböck et al.) [(paper)](http://arxiv.org/pdf/1612.00686v1.pdf) 
* 2016 - Reproducibility of Retinal Thickness Measurements across Spectral-Domain Optical Coherence Tomography Devices using Iowa Reference Algorithm (Rashid et al.) [(paper)](http://arxiv.org/pdf/1612.06442v1.pdf) 
* 2015 - 3D Automatic Segmentation Method for Retinal Optical Coherence Tomography Volume Data Using Boundary Surface Enhancement (Sun et al.) [(paper)](http://arxiv.org/pdf/1508.00966v1.pdf) 
* 2015 - Accurate automatic segmentation of retina layers with emphasis on first layer (Salarian) [(paper)](http://arxiv.org/pdf/1501.06114v2.pdf) 
* 2014 - Probabilistic Intra-Retinal Layer Segmentation in 3-D OCT Images Using Global Shape Regularization (Rathke et al.) [(paper)](http://arxiv.org/pdf/1403.8003v1.pdf) 
* 2014 - State-of-the-Art in Retinal Optical Coherence Tomography Image Analysis (Baghaie et al.) [(paper)](http://arxiv.org/pdf/1411.0740v2.pdf) 
* 2013 - Thickness Mapping of Eleven Retinal Layers in Normal Eyes Using Spectral Domain Optical Coherence Tomography (Kafieh et al.) [(paper)](http://arxiv.org/pdf/1312.3199v1.pdf) 
* 2012 - A 3D Segmentation Method for Retinal Optical Coherence Tomography Volume Data (Sun et al.) [(paper)](http://arxiv.org/pdf/1204.6385v1.pdf) 
* 2012 - Intra-Retinal Layer Segmentation of 3D Optical Coherence Tomography Using Coarse Grained Diffusion Map (Kafieh et al.) [(paper)](http://arxiv.org/pdf/1210.0310v2.pdf) 
<br/><br/>


