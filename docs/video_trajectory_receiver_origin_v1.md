# Saved G receiver-origin diagnostic

This fixed user-run diagnostic reads the saved FULL181 MP4-readback RGB8 P0/P1 files from G run 20261005T140121854219Z. It verifies both SHA256 identities before slicing [0:177] and [1:178]; start1 must also match the old crop SHA. There is no codec fallback, new source, framewise VAE, writer, generation, sync search, offset correction, or parameter menu.

Each received clip gets one frozen Wan VAE FP32 posterior-mode encode, reused for K0/K1 with the unchanged reader and R44. Fixed denominator: 4 encodes, 8 reads, 256 final bits, 11264 time-bit rows, 337920 individual votes (1320 per bit, 42240 per read). Failed reads keep explicit artifact/status slots and planned denominators.

Blind gzip records factor votes as [payload channel, receiver latent index 1..44, original frequency-coordinate-list index], with exact coordinate order and a coefficient-zero mask. Signed vote is 2*int(FFT.real>0)-1; exact zero remains a zero vote (-1 signed), not erasure. A bit follows votes[ch,:,bit::8].reshape(-1) in C order, preserving the old Counter first-encounter tie. The adapter verifies all votes and decoded bits against the unchanged reader. Its additional FFT only records the same votes, not another payload-method call.

Receiver indices and nominal stride4 coordinates are indexing labels, not independent four-frame supports or definite source-frame ownership. The reader receives only the normalized clip and key. Condition, crop start, expected message, truth-signed margins and historical start1 comparisons join after blind sealing. Historical differences never trigger retries or selection.

The draft notebooks/video_trajectory_receiver_origin_v1_colab.ipynb has five code cells and the exact first Drive mount. SOURCE_SHA=None stops before creating output directories. After review and source publication, the upper reviewer binds the immutable SHA:

    python -B scripts/build_video_trajectory_receiver_origin_notebook.py --source-sha <published-source-SHA>

Run all is for the user after binding. Only CPU FFT, fake backend and static checks ran here; real Wan execution, Drive access and Colab persistence remain unexecuted.

Relevant versions follow saved G: torch public2.11.0 (record actual CUDA/local build), diffusers0.39.0, transformers4.57.6, numpy2.1.3, accelerate1.15.0, safetensors0.8.0, huggingface-hub0.36.2, tokenizers0.22.2. One necessary fixed repair is allowed before running; unrelated pip-check warnings are recorded. The notebook preserves process-group cleanup/failure persistence, probes Wan VAE only, and has no GPU model-name gate. Catchable interruption releases the VAE and seals failed slots.

result.json retains source/config/environment, actual/planned counts and all final bits. blind/ contains complete votes, posthoc/ contains time-bit truth joins, and blind_receiver_readouts.json binds the pre-truth snapshot. Input/history identities are fixed in experiments/wan_state_clock/configs/video_trajectory_receiver_origin_v1.json.

Local validation:

    /home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python -B -m pytest -q tests/test_video_trajectory_receiver_origin_v1.py -p no:cacheprovider

Adjacent origins also change the content window. This bounded compatibility diagnostic cannot uniquely prove phase causality, establish offset recovery, calibrated detection/FPR, generalization, or scientific PASS.

