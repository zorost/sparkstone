# 07 · Legal, licensing and credits

> This is a practical summary written for planning, not legal advice. Before commercial launch, have a lawyer
> review the licence choice and run a trademark clearance search for "Sparkstone" in your markets.

## 1. What we are allowed to do

| Component | Owner | Licence | What it allows | What it requires |
|---|---|---|---|---|
| **Basalt** (engine) | Jesús Gulfo | MIT | Use, modify, redistribute, **sell**, sublicense | Keep the copyright notice and licence text in copies |
| Strata (Basalt's base) | Niko1221 | MIT | same | same |
| Infernix (kernels adapted in Basalt) | Ian Ranson et al. | Apache-2.0 | same, plus a patent grant | Keep its licence/NOTICE; mark changed files |
| ggml / llama.cpp (extracted in Basalt) | ggml authors | MIT | same | Keep notice |
| **Inkling 2 Mini** weights | Thinking Machines Lab | Apache-2.0 | Run, fine-tune, serve commercially | Keep notices if you redistribute the weights |
| Basalt's packs on Hugging Face | conversions by Basalt's author | each model's own licence | — | Users download them; **we do not redistribute weights** |
| NVIDIA CUDA runtime | NVIDIA | CUDA EULA | Link and redistribute the runtime libraries under the EULA | We ship no CUDA libraries; users install CUDA |

Conclusion: **building, publishing and commercializing Sparkstone is permitted**, as long as notices are kept and
nobody is implied to endorse us. Basalt's upstream sources stay MIT (we distribute patches and build scripts, not a
relicensed copy).

## 2. Our licensing choice, and why

| Part | Licence | Reason |
|---|---|---|
| Sparkstone's code and docs (router, tools, scripts, configs, docs) | **Apache-2.0** | Permissive, so companies adopt it freely, which builds reputation. It also gives an explicit patent grant, and **the NOTICE file must travel with every redistribution** (§4(d)), so the credit to Zoros is a legal condition, not a courtesy |
| Engine patches (`patches/`) | **MIT** | Same terms as Basalt, so they can be merged upstream without friction. Getting them merged is itself good for reputation ("Zoros contributed DGX Spark support to Basalt") |
| Name and logos | **Trademark, all rights reserved** | This is where reputation and commercial value actually sit. Code can be forked; the name cannot be used without permission ([TRADEMARKS.md](../TRADEMARKS.md)) |
| Citation | `CITATION.cff` | GitHub shows "Cite this repository". Researchers and benchmark authors reference you properly |

### "Can we require people to reference us?"

Yes, in three layers that work together:

1. **Legally binding:** Apache-2.0 §4(d). Anyone redistributing Sparkstone (or software containing it) must keep the
   NOTICE text, including the Zoros line. Removing it ends their licence.
2. **Brand-binding:** TRADEMARKS.md. Forks may not use the Sparkstone name, so derivatives must either credit
   "a fork of Sparkstone" or rebrand entirely. Either way the original stays clearly yours.
3. **Social:** CITATION.cff, the "Built with Sparkstone" badge, and a request in the README.

What a permissive licence **cannot** do is force attribution in someone's product UI, or stop a company from
offering Sparkstone as a hosted service. If that matters, the alternatives are:

| Option | Effect | Cost |
|---|---|---|
| Apache-2.0 + trademark (**recommended**) | Maximum adoption and reputation; brand protected | Competitors may use the code (with credit) |
| AGPL-3.0 + commercial licence (dual licensing) | Network use must publish modifications; sell exceptions | Many companies avoid AGPL outright. Also, only *our* code would be AGPL, since Basalt underneath stays MIT, which weakens the lever |
| Business Source Licence (source-available, converts to open after N years) | Blocks competing commercial offerings | Not "open source"; the community trusts it less, which hurts virality |

For a company whose goal is **reputation first, revenue from expertise**, Apache-2.0 + trademark is the standard
choice (Kubernetes, Ray and vLLM's ecosystem all follow it).

## 3. How Zoros can commercialize (all compatible with the above)

- **Support and SLAs** for teams running Spark fleets: install, tuning, upgrades, on-call.
- **Signed prebuilt releases and containers** for DGX OS: convenience plus trust. The name guarantees the source.
- **Twin and cluster appliance packages:** cabling guide, router as a service, monitoring dashboards.
- **Custom model packs and tuning:** quantizing and validating customer models for GB10.
- **Consulting:** on-prem and air-gapped LLM deployment ("sovereign AI on your desk").
- **Training and workshops.**

Any proprietary add-ons (a management UI, a fleet controller) can live in a separate repository under a commercial
licence. Apache-2.0 allows that.

## 4. Obligations checklist (do these, keep doing them)

- [x] Keep Basalt's MIT notice: patches carry it (`patches/LICENSE`); the engine tree keeps upstream's LICENSE
- [x] NOTICE lists Basalt and its third parties, models and trademarks
- [x] No model weights in the repository; users download them under each model's licence
- [x] No NVIDIA trademarks in our logo; "for DGX Spark" used only descriptively, with a non-affiliation statement
- [x] No claim of endorsement by NVIDIA, the Basalt project or Thinking Machines Lab
- [ ] **Tell Basalt's author before launch** and offer the patches upstream (goodwill, and the ecosystem norm)
- [ ] Trademark clearance search, then file "SPARKSTONE" (USPTO classes 9 and 42; EUIPO/UKIPO if relevant)
- [ ] Register the GitHub org, the Hugging Face org, the X handle and the domain/subpath before announcing
- [ ] If you ever ship prebuilt binaries: include Basalt's LICENSE and THIRD_PARTY_LICENSES inside the archive
      (upstream's `tools/package_release.py` already does this)
- [ ] If you ever redistribute CUDA libraries in a container: use NVIDIA's CUDA base images (EULA-compliant)

## 5. Credits (keep this list in every public description)

- **Basalt**: Jesús Gulfo and contributors: the engine, kernels, drafters, server ([GitHub](https://github.com/jesdga95/basalt), [sponsor](https://github.com/sponsors/jesdga95))
- **Strata** (Niko1221), **Infernix** (Ian Ranson and contributors), **ggml / llama.cpp**, **NInfer**
- **Inkling 2 Mini**: Thinking Machines Lab
- **DGX Spark community measurements** cited in [research](research/), especially NVIDIA's open
  `dgx-spark-playbooks` and the public vLLM/SGLang Inkling-on-Spark benchmark repositories
