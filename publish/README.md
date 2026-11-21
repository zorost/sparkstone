# Launch kit

| File | Where it goes |
|---|---|
| [`x-thread.md`](x-thread.md) | X / Twitter (7 posts; social card attached to post 1) |
| [`github-release.md`](github-release.md) | GitHub release notes for the tag |
| [`huggingface/`](huggingface) | A Hugging Face repo `zoros/Inkling-2-Mini-Sparkstone` (card + Spark configs; no weights) |
| [`website/`](website) | Drop-in page for zoros.com/sparkstone (one HTML file + 2 images, no external dependencies) |
| [`upstream-pr.md`](upstream-pr.md) | Message and PR plan for Basalt's author |
| `../assets/brand/png/social-card-1200x630.png` | Open Graph / X card / LinkedIn image |

## Launch checklist

**Before anything public**
- [ ] Confirm company name, domain, GitHub org, HF org, X handle and contact email. The placeholders used are
      `Zoros`, `zoros.com`, `huggingface.co/zoros`, `hello@zoros.com`
      (`grep -rn "zoros" .` lists every use)
- [ ] Reserve the GitHub org, the HF org and the X handle; run a trademark clearance search for "Sparkstone"
- [ ] Message Basalt's author ([upstream-pr.md](upstream-pr.md))

**Proof (v0.2)**
- [ ] `FULL=1 scripts/selftest.sh` on Spark 1 and Spark 2: stage 4 (GPU parity) green
- [ ] Quality gate (greedy identity, PPL/KLD) passes
- [ ] Benchmark rows + vLLM/SGLang baselines on the same box ([docs/05](../docs/05-benchmarking.md))
- [ ] Twin: `check-link.sh` report; data-parallel aggregate throughput at 8 streams
- [ ] Replace every "projection" / `{{...}}` in README, website, X thread and HF card with measured numbers

**Publish**
- [ ] Push the repo (public), tag `v0.2.0`, release notes, add a GitHub topic set (`dgx-spark`, `gb10`, `nvfp4`, `llm-inference`)
- [ ] HF repo with configs + results
- [ ] Website page live, then the X thread linking to GitHub and the page
- [ ] Post on NVIDIA's DGX Spark developer forum and r/LocalLLaMA with the raw self-test report attached
- [ ] Open the upstream PRs
