#!/usr/bin/env bash
# SDC3b reproduction: train emulators → run inference → plot
# Pk emulator: seed 51  |  xHI emulator: seed 34 (best match to legacy xHI posteriors)
set -e

cd "$(dirname "$0")/.."

echo "=== [1/5] Training Pk emulator (seed 51) ==="
python train.py -opt options/emulators/Pk_minmax_extended_SDC3b.yml
echo "    Saved to: trained_emulators/Pk_SDC3b_MLP_minmax_extended/"

echo "=== [2/5] Training xHI emulator (seed 34) ==="
python train.py -opt options/emulators/xHI_SDC3b.yml
echo "    Saved to: trained_emulators/xHI_SDC3b_minmax/"

echo "=== [3/5] Running PS1 inference ==="
python inference.py -opt options/inference/sdc3b_PS1.yml
echo "    Saved to: inferences/SDC3b_PS1/"

echo "=== [4/5] Running PS2 inference ==="
python inference.py -opt options/inference/sdc3b_PS2.yml
echo "    Saved to: inferences/SDC3b_PS2/"

echo "=== [5/5] Generating plots ==="
# single reproduced chain
python analysis/plot_triangle_sdc3b.py --PS=PS1 --chains inferences/SDC3b_PS1/LikelihoodSDC3b --labels Reproduced
python analysis/plot_triangle_sdc3b.py --PS=PS2 --chains inferences/SDC3b_PS2/LikelihoodSDC3b --labels Reproduced
python analysis/SDC3b_xHI.py --PS=PS1 --chains inferences/SDC3b_PS1/LikelihoodSDC3b --labels Reproduced
python analysis/SDC3b_xHI.py --PS=PS2 --chains inferences/SDC3b_PS2/LikelihoodSDC3b --labels Reproduced

# legacy vs reproduced overlay
python analysis/plot_triangle_sdc3b.py --PS=PS1 --chains scripts/non-public/LikelihoodSDC3b_SDC3b_PS1 inferences/SDC3b_PS1/LikelihoodSDC3b --labels Legacy Reproduced
python analysis/plot_triangle_sdc3b.py --PS=PS2 --chains scripts/non-public/LikelihoodSDC3b_SDC3b_PS2 inferences/SDC3b_PS2/LikelihoodSDC3b --labels Legacy Reproduced
python analysis/SDC3b_xHI.py --PS=PS1 --chains scripts/non-public/LikelihoodSDC3b_SDC3b_PS1 inferences/SDC3b_PS1/LikelihoodSDC3b --labels Legacy Reproduced
python analysis/SDC3b_xHI.py --PS=PS2 --chains scripts/non-public/LikelihoodSDC3b_SDC3b_PS2 inferences/SDC3b_PS2/LikelihoodSDC3b --labels Legacy Reproduced
echo "    Saved to: analysis/"

echo "=== Done ==="
