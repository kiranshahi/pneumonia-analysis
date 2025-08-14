
import argparse
from pathlib import Path
import subprocess
import sys

ARCHS = ["resnet18","densenet121","efficientnet_b0"]

def main():
    parser = argparse.ArgumentParser(description="Train multiple architectures sequentially")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--archs", type=str, default=",".join(ARCHS))
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    for arch in archs:
        out_dir = Path(args.out_root) / arch
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "pneumonia_cnn.train",
            "--data_dir", args.data_dir,
            "--out_dir", str(out_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--seed", str(args.seed),
            "--img_size", str(args.img_size),
            "--arch", arch
        ]
        if args.no_pretrained:
            cmd.append("--no_pretrained")
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
