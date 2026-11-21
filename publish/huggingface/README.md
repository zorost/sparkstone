---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
tags:
  - dgx-spark
  - gb10
  - nvfp4
  - basalt
  - sparkstone
  - inference
  - moe
pipeline_tag: image-text-to-text
---

<p align="center"><img src="https://raw.githubusercontent.com/zorost/sparkstone/main/assets/brand/png/sparkstone-lockup-light.png" width="480" alt="Sparkstone"></p>

# Inkling 2 Mini · Sparkstone (DGX Spark)

Run **Thinking Machines' Inkling 2 Mini** on an **NVIDIA DGX Spark** with [Sparkstone](https://github.com/zorost/sparkstone),
the Basalt engine ported to GB10.

This repository holds **Spark configurations and benchmark results**, not new weights. The model file is Basalt's
pack [`jesdga/Inkling-2-Mini-Basalt`](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt)
(`Inkling-2-Mini-NVFP4-q8.basalt`, 19.7 GB). It is linked rather than copied, so it stays in sync with upstream
and credit stays with its author.

## Quick start (on the Spark)

```bash
git clone https://github.com/zorost/sparkstone && cd sparkstone
scripts/build-spark.sh && scripts/download-model.sh && scripts/serve-spark.sh
```

## Files

| File | |
|---|---|
| `basalt_config.spark.json` | 256K context, FP8 KV |
| `basalt_config.spark-1m.json` | 1,048,576 context, FP8 KV (~16 GB KV) |
| `results/` | Self-test reports from real DGX Spark units (added as they are measured) |

## Results

| Setup | conv tok/s | struct tok/s | prose tok/s | 16K prefill |
|---|---:|---:|---:|---:|
| 1× DGX Spark | pending | pending | pending | pending |
| 2× DGX Spark (Twin, aggregate) | pending | pending | pending | pending |

Method: Basalt's `basalt-bench --community` preset. See
[benchmarking](https://github.com/zorost/sparkstone/blob/main/docs/05-benchmarking.md).

## Credits and licence

Model: Inkling 2 Mini © Thinking Machines Lab, Apache-2.0. Pack and engine: [Basalt](https://github.com/jesdga95/basalt)
by Jesús Gulfo, MIT. Configs and results here: Apache-2.0, © Zoros. Not affiliated with or endorsed by NVIDIA,
Thinking Machines Lab or the Basalt project.
