# HMSA-DTI
HMSA-DTI: The codes demo for paper "Hierarchical Multimodal Self-Attention based Graph Neural Network for DTI Prediction".

# Required Packages
View requirements.txt and environment.yml

# Datset
Please make sure dataset format is csv, and column name are: 'smiles','sequences','label'. We have provided dataset in the dataset directory, which can also be obtained from the following link.<br>
1. DrugBank is available at [https://github.com/MrZQAQ/MCANet](https://github.com/MrZQAQ/MCANet) <br>
2. Human and C.elegans are available at [https://github.com/Layne-Huang/CoaDTI](https://github.com/Layne-Huang/CoaDTI)
3. BioSNAP is available at [https://snap.stanford.edu/biodata/datasets/10002/10002-ChG-Miner.html](https://snap.stanford.edu/biodata/datasets/10002/10002-ChG-Miner.html)

# Quick start
You can use the following code to run our demo.<br>
python train.py --data_path ./dataset/DrugBank.csv --metric auc --dataset_type classification --save_dir train_results --target_columns label --epochs 150 --ensemble_size 1 --num_folds 10 --batch_size 50 --aggregation mean --dropout 0.1 --save_preds

