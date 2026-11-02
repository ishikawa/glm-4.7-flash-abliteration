# Direction artifacts

These two archives allow editing without repeating activation collection and
fitting. They use the original GLM-4.7-Flash activations from the four frozen
splits in `data/direction/`.

| File | SHA-256 |
| --- | --- |
| `directions.npz` | `e249101ac527ca3751de74e9b6ff7398c2e9c17d6bc225c05c18aee06a0ddfb4` |
| `benign_means.npz` | `5df47cdb3ac0fe46d0d93d317fd4f6326542cbe52aa2767d44eea3fb6319bd45` |

Hashes were calculated with `shasum -a 256 artifacts/*.npz`. Both archives contain
the following float32 arrays, with zero-based layer indices:

| Array | Shape | Meaning |
| --- | --- | --- |
| `attn_out` | `[47, 2048]` | Attention output site |
| `ffn_out` | `[47, 2048]` | FFN output site |
| `resid_mean` | `[47, 2048]` | Residual averaged across its single stream |

In `directions.npz`, each row is the unit-normalized harmful fit mean minus
benign fit mean. In `benign_means.npz`, each row is the unnormalized benign fit
mean used for biprojection. Held-out records contribute only to validation
statistics, not either fit mean. No massive-activation mask is enabled.

After collecting each split into the directories below, regenerate with:

```sh
python scripts/direction_fit.py \
  --harmful _work/activations/harmful-fit \
  --benign _work/activations/benign-fit \
  --held-harmful _work/activations/harmful-heldout \
  --held-benign _work/activations/benign-heldout \
  --output _work/directions.npz \
  --benign-means-out _work/benign_means.npz \
  --report _work/direction-report.json
```

This command expresses the confirmed input splits and fit settings with local
output paths. The original activation manifests identify 200 harmful and 200
benign fit records, with 50 records in each held-out split. The supplied arrays
were checked for exact equality against a CPU rerun of this command using those
activations. Whether the benign means archive was first written alongside the
directions or in a later fit invocation is not recorded.

The selected editing direction is `resid_mean[24]`. The original-model collection
used the quantized baseline, profile `glm47`, context 8192, microbatch 2048 and
eight CPU threads, with the embedded chat template and Unix epoch template time.
See [the collector](../tools/activation-collect/README.md) for build and collection
instructions and [the frozen data](../data/README.md) for split provenance.
