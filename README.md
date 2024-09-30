# BalDistill

**Paper**:  
Yuhang Zhou, Jing Zhu, Paiheng Xu, Xiaoyu Liu, Xiyao Wang, Danai Koutra, Wei Ai, Furong Huang  
*Multi-Stage Balanced Distillation: Addressing Long-Tail Challenges in Sequence-Level Knowledge Distillation*  
[Link to Paper](https://arxiv.org/abs/2406.13114)

### Citation (BibTeX):
```bibtex
@misc{zhou2024multistagebalanceddistillationaddressing,
      title={Multi-Stage Balanced Distillation: Addressing Long-Tail Challenges in Sequence-Level Knowledge Distillation}, 
      author={Yuhang Zhou and Jing Zhu and Paiheng Xu and Xiaoyu Liu and Xiyao Wang and Danai Koutra and Wei Ai and Furong Huang},
      year={2024},
      eprint={2406.13114},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.13114}, 
}
```

## Usage

First, navigate to the directory:

```bash
cd LLM-Finetuning-Hub/llama2/
```

### Training:

```bash
python llama2_classification_active.py --pretrained_ckpt {llm_checkpoint} --dataset_dir ../../dataset/abstractive-qa/ --dataset abstractive-qa_cot_active --method random --epochs 8 --budget {budget_number}
python llama2_classification_active.py --pretrained_ckpt {llm_checkpoint} --dataset_dir ../../dataset/abstractive-qa/ --dataset abstractive-qa_cot_active --method balanced --epochs 8 --budget {budget_number}
python llama2_classification_active.py --pretrained_ckpt {llm_checkpoint} --dataset_dir ../../dataset/abstractive-qa/ --dataset abstractive-qa_cot_active --method adaptive --epochs 8 --budget {budget_number}
```

### Inference:

```bash
python llama2_classification_inference.py --experiment_dir experiments/active_learning/{fine-tuned_checkpoint} --dataset abstractive-qa --dataset_dir ../../dataset/abstractive-qa/
```

### Parameters:

- **{llm_checkpoint}**: Checkpoint to store the Hugging Face format Llama2 or Llama3
- **{budget_number}**: The budget number for each domain. For abstractive-qa, it is 1,000 or 2,000
- **{fine-tuned_checkpoint}**: After fine-tuning, a checkpoint folder will appear in `experiments/active_learning/`
