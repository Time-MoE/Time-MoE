

path="/share/dean/embeds/"

# Run the 4 benchmark processes in parallel with separate log files
CUDA_VISIBLE_DEVICES=0 python bench_github.py --ablation --input_horizon 100 --output_dir "${path}embed_github_pushes" 2>&1 | tee "embed_github_pushes.log"&
CUDA_VISIBLE_DEVICES=1 python bench_github.py --input_horizon 100 --output_dir "${path}embed_github" 2>&1 | tee "embed_github.log" &
CUDA_VISIBLE_DEVICES=2 python bench_arxiv.py --ablation --input_horizon 100 --output_dir "${path}embed_arxiv_accesses" 2>&1 | tee "embed_arxiv_accesses.log" &
CUDA_VISIBLE_DEVICES=3 python bench_arxiv.py --input_horizon 100 --output_dir "${path}embed_arxiv" 2>&1 | tee "embed_arxiv.log" &

# # Wait for all background processes to complete
# wait

# Force ALL temporary operations to use shared storage
export TMPDIR=/share/dean/temp_processing
export TMP=/share/dean/temp_processing  
export TEMP=/share/dean/temp_processing
export HF_DATASETS_CACHE=/share/dean/hf_cache
export HF_HOME=/share/dean/hf_home

# Create the directories
mkdir -p /share/dean/temp_processing
mkdir -p /share/dean/hf_cache
mkdir -p /share/dean/hf_home

# Now run your script
python concat_embed.py --path /share/dean/embeds/

echo "All benchmark processes completed. Starting concat operations..."

# Run the concat operations sequentially with separate log files
python concat_embed.py --path "${path}embed_github_pushes" 2>&1 | tee "concat_github_pushes.log"
python concat_embed.py --path "${path}embed_github" 2>&1 | tee "concat_github.log"
python concat_embed.py --path "${path}embed_arxiv_accesses" 2>&1 | tee "concat_arxiv_accesses.log"
python concat_embed.py --path "${path}embed_arxiv" 2>&1 | tee "concat_arxiv.log"

echo "Job completed successfully!"


