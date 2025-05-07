import argparse
from time_moe.runner import TimeMoeRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        help="Path to training data. (Folder contains data files, or data file)",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="Maple728/TimeMoE-50M",
        help="Path to pretrained model. Default: Maple728/TimeMoE-50M",
    )
    parser.add_argument("--output_path", "-o", type=str, default="logs/time_moe")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for training. Default: 1024",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Step size for sliding the time-series window. Defaults to the value of max_length if not specified.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "--min_learning_rate", type=float, default=5e-5, help="minimum learning rate"
    )

    parser.add_argument(
        "--train_steps", type=int, default=None, help="number of training steps"
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=1.0, help="number of training epochs"
    )
    parser.add_argument(
        "--normalization_method",
        type=str,
        choices=["none", "zero", "max"],
        default="zero",
        help="normalization method for sequence",
    )

    parser.add_argument("--seed", type=int, default=9899, help="random seed")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["auto", "eager", "flash_attention_2"],
        default="auto",
        help="attention implementation",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        choices=["constant", "linear", "cosine", "constant_with_warmup"],
        default="cosine",
        help="learning rate scheduler type",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="warmup ratio")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")

    parser.add_argument(
        "--global_batch_size", type=int, default=64, help="global batch size"
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=16, help="micro batch size per device"
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        type=str,
        default="fp32",
        help="precision mode (default: fp32)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="enable gradient checkpointing",
    )
    parser.add_argument(
        "--deepspeed", type=str, default=None, help="DeepSpeed config file path"
    )

    parser.add_argument(
        "--from_scratch", action="store_true", help="train from scratch"
    )
    parser.add_argument(
        "--save_steps", type=int, default=None, help="number of steps to save model"
    )
    parser.add_argument(
        "--save_strategy",
        choices=["steps", "epoch", "no"],
        type=str,
        default="no",
        help="save strategy",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="limit the number of checkpoints",
    )
    parser.add_argument(
        "--save_only_model", action="store_true", help="save only model"
    )

    parser.add_argument(
        "--logging_steps", type=int, default=1, help="number of steps to log"
    )
    parser.add_argument(
        "--evaluation_strategy",
        choices=["steps", "epoch", "no"],
        type=str,
        default="no",
        help="evaluation strategy",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=None, help="number of evaluation steps"
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="max gradient norm"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="number of workers for dataloader",
    )

    args = parser.parse_args()

    if args.normalization_method == "none":
        args.normalization_method = None

    runner = TimeMoeRunner(
        model_path=args.model_path,
        output_path=args.output_path,
        seed=args.seed,
    )

    runner.train_model(
        from_scratch=args.from_scratch,
        max_length=args.max_length,
        stride=args.stride,
        data_path=args.data_path,
        normalization_method=args.normalization_method,
        attn_implementation=args.attn_implementation,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        train_steps=args.train_steps,
        num_train_epochs=args.num_train_epochs,
        precision=args.precision,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        save_only_model=args.save_only_model,
        save_total_limit=args.save_total_limit,
    )
