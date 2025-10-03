
KNN ablation done on Time-MoE embeddings for datasets released in Lead-Lag forecasting for social platforms: ArXiv and Github


to run this, you need to run a script similar to `run.sh` to generate embeddings for your dataset. After the script is done, you could run `knn2.py --case ${case}` with desired case.

the correct paths get auto-generated and are setup throughout the script, you just need to change the path for your data, cache and save directory in `run.sh`