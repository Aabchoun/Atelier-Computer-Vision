# Fire Spread Simulator & Predictor

A multi-agent computer vision system for vegetated areas (Sup'Com computer vision project). Perception agents detect the fire, smoke, fuel and burned area on satellite data. A deep network predicts how the fire front evolves. A cellular automaton simulates the same spread. A coordinator fuses everything on **one scene and one grid** into a risk map, uncertain zones and alerts.

**Current mode: local only.** No cloud service is needed yet (no Earth Engine, Redis or Ray). The agents talk through plain Python calls. A synthetic scene lets the whole pipeline run before any real data is downloaded.

## Quick start (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python scripts/make_synthetic_scene.py                       # -> data/scenes/synthetic/
python scripts/run_pipeline.py --scene data/scenes/synthetic # -> outputs/synthetic/ (report.md, quicklook.png, datacube/)
pytest                                                       # tests must stay green
```

Optional extras: `pip install -e ".[ml]"` (PyTorch + segmentation_models_pytorch, for training), `".[geo]"` (rasterio, pyproj, to read real GeoTIFF / FIRMS exports), `".[gee]"` (Earth Engine API), `".[app]"` (Streamlit dashboard: `streamlit run app/streamlit_app.py`).

## Personal access (FIRMS, Earth Engine)

Each member keeps their own keys in a `.env` file at the project root. Git ignores this file; the template is `.env.example`.

```
FIRMS_MAP_KEY=<your key>
GEE_PROJECT=<your Earth Engine Cloud project id>
```

```powershell
pip install -e ".[geo,gee]"
earthengine authenticate              # once per machine, opens the browser
python scripts/check_access.py        # both lines must say OK
python scripts/fetch_firms.py --bbox 8.4 36.6 9.2 37.1 --start 2023-07-17 --end 2023-07-31
```

Never put a key in code, configs, notebooks, screenshots or commits. If a key leaks, request a new one.

## Real scenes (Earth Engine + FIRMS)

The case-study fire is **Evros, Greece, August 2023**. [docs/case_study.md](docs/case_study.md) explains why it was chosen and gives the first results.

```powershell
# list Sentinel-2 acquisitions (coverage / clouds) before choosing a day
python scripts/build_scene.py --name evros --bbox 25.65 40.74 26.50 41.32 --fire-start 2023-08-19 --date 2023-08-23 --list-s2
# build the scene for forecast day t (about 3-8 min), then run the system on it
python scripts/build_scene.py --name evros_2023-08-23 --bbox 25.65 40.74 26.50 41.32 --fire-start 2023-08-19 --date 2023-08-23
python scripts/run_pipeline.py --scene data/scenes/evros_2023-08-23
```

Every layer is fetched directly onto the common grid (UTM, 375 m; optical bands at 375/16 = 23.4 m). The ground truth is the set of FIRMS hotspots on day t+1.

### Day-by-day evaluation and CA calibration

```powershell
# one scene per day, downloads shared and cached in data/sequences/evros_2023/shared (re-run resumes)
python scripts/build_sequence.py --name evros_2023 --bbox 25.65 40.74 26.50 41.32 --fire-start 2023-08-19 --start 2023-08-19 --end 2023-09-02
# run the coordinator day after day (one Coordinator: fallback layers and fire history carry over)
python scripts/run_sequence.py --sequence data/sequences/evros_2023 --split-date 2023-08-25
# calibrate the cellular automaton on days <= 08-25, test on the later days, write configs/evros_2023_calibrated.yaml
python scripts/calibrate_ca.py --sequence data/sequences/evros_2023 --runs outputs/evros_2023 --train-end 2023-08-25
python scripts/run_sequence.py --sequence data/sequences/evros_2023 --split-date 2023-08-25 --config configs/evros_2023_calibrated.yaml --out outputs/evros_2023_calibrated
```

Days without a Sentinel-2 pass get no `s2_post`. The fire then comes from FIRMS only, while smoke and the burned area fall back on their last known layers. Results are written to `outputs/<sequence>/summary.md` and `summary.png`. Metrics are reported exact and with a 1-cell tolerance (`F1@1`).

## How it works

```
RawScene (one AOI, one time window, each raster at native resolution)
  Sentinel-2 pre/post · FIRMS · ERA5 · DEM · WorldCover
        │
        ▼
COORDINATOR ── runs the perception agents in order:
  fire ──► fusion of fire_s2 + fire_firms (confidence × staleness, NaN = cloud) ──► fire
  smoke · vegetation · burned_area · environment
  every output is resampled onto the common grid and written to the
        │
        ▼
SHARED SCENE DATACUBE (blackboard, same CRS / extent / resolution)
        │
        ├──► spread agent       (persistence baseline, later U-Net / ConvLSTM) ─► spread_prob
        └──► simulation agent   (Alexandridis cellular automaton, Monte Carlo) ─► sim_prob
        │
        ▼
ensemble ─► risk, disagreement, uncertain zones, alerts, wind check, report
```

Robustness is built in. If an agent crashes, or its scene is too cloudy, the coordinator falls back on the last known layer, marks it `stale`, and the run continues. Disabling an agent (`--disable vegetation`) is the ablation study.

## Repository layout

```
configs/default.yaml       all parameters (thresholds, fusion, CA coefficients, enabled agents)
firespread/
  common/                  message format, Grid + resampling, RawScene / SceneDatacube,
                           base Agent, metrics, spectral indices, U-Net + loss + checkpoints (nn.py)
  agents/
    fire/                  SWIR active fire + FIRMS
    smoke/                 smoke mask + plume direction
    vegetation/            land cover, NDVI, fuel (p_veg, p_den, burnable)
    burned_area/           dNBR burn scar
    environment/           ERA5 wind / humidity / temperature, DEM slope / aspect
    spread/                spread prediction (persistence -> U-Net -> ConvLSTM)
    simulation/            cellular automaton (cell agents) + calibration
  coordinator/             orchestration, fusion, fallback, alerts, report
  io/                      scene format, FIRMS CSV reader, synthetic scene
  evaluation.py            benchmark table (precision, recall, F1, IoU, AUC-PR)
  viz.py                   quick-look figure
scripts/                   make_synthetic_scene, run_pipeline, run_ablation
app/                       Streamlit dashboard (phase 5)
tests/                     pytest suite
data/  models/  outputs/   local only, not versioned (see data/README.md)
notebooks/                 exploration notebooks
```

## The contract every agent follows

```python
class MyAgent(Agent):
    name = "my_agent"
    produces = ("my_layer",)

    def run(self, raw: RawScene, cube: SceneDatacube) -> list[AgentMessage]:
        ...
        return [self.message(raw.grid, "my_layer", array, acquisition_time=..., sensor=..., confidence=..., cloud_fraction=...)]
```

Each message carries the common format `{agent, layer, acquisition_time, sensor, crs, bbox, resolution, data_ref, confidence, cloud_fraction}` plus `stale` and `extras`.

- Perception agents may return arrays at **any resolution covering the scene bbox**: the coordinator resamples them (method per layer in `coordinator.resampling`).
- Use `NaN` for "not observed" (clouds). Never use 0, because 0 means "observed, nothing there".
- Raise `AgentInputError` when an input is missing. The coordinator handles the fallback.
- Every agent has a **baseline that needs no training**. Setting `checkpoint:` in its config section switches it to your trained model. The checkpoint format is described in `firespread/common/nn.py` and includes the ordered list of input bands/layers and the normalisation statistics.

## Layers in the datacube

| layer | produced by | unit / meaning |
|---|---|---|
| fire_s2, fire_firms | fire | active-fire probability (optical, NaN under clouds) / FIRMS confidence |
| fire | coordinator | fused active-fire probability |
| smoke | smoke | 0..1 smoke score (`extras.plume_direction_deg`) |
| ndvi, landcover, burnable, p_veg, p_den | vegetation | NDVI, WorldCover class, 0/1, Alexandridis fuel terms |
| dnbr, burned | burned_area | dNBR, burned fraction |
| wind_u, wind_v, t2m, rh | environment | m/s (towards east/north), degC, % |
| elevation, slope, aspect | environment | m, deg, deg (downslope azimuth) |
| spread_prob | spread | P(active fire on day t+h) |
| sim_prob, sim_reach_prob | simulation | P(active fire on day t+h), P(burned by t+h) |
| risk, disagreement, uncertain | coordinator | ensemble, abs(spread - sim), 0/1 flag |

## Who works where (adapt to the team)

| owner | folder | phase 1 task | trained with |
|---|---|---|---|
| A | `agents/spread/` | persistence → U-Net → ConvLSTM | Next Day Wildfire Spread, WildfireSpreadTS |
| B | `agents/fire/` | SWIR U-Net | Landsat-8 Active Fire, Sen2Fire |
| C | `agents/smoke/` | smoke U-Net + **own labels** | USTC_SmokeRS, Sen2Fire aerosol, own S2 masks |
| D | `agents/vegetation/`, `agents/burned_area/` | land-cover classifier, burn-scar U-Net | EuroSAT, WorldCover, CaBuAr, EFFIS |
| E | `agents/environment/`, `agents/simulation/`, `io/` | CA calibration, scene builder | burned areas, ERA5, DEM |

Put your dataset loader and training script next to your agent (`agents/<name>/dataset.py`, `train.py`). Put datasets in `data/raw/<dataset>/` and checkpoints in `models/`.

## Experiments

```powershell
python scripts/run_ablation.py --scene data/scenes/synthetic --layer sim_prob    # remove one agent at a time
python scripts/make_synthetic_scene.py --cloud-cover 0.4 --out data/scenes/synthetic_cloudy
python scripts/run_pipeline.py --scene data/scenes/synthetic_cloudy             # robustness: FIRMS takes over
```

Metrics on the synthetic scene only check the plumbing. They say nothing about real performance.

## Rules of the project

- Report precision, recall, F1, IoU and AUC-PR, never plain accuracy. Always include the **persistence baseline**.
- Mask `-1` (missing) labels out of the loss and the metrics (`masked_focal_dice_loss`, `metrics.py`).
- Split train/val/test **by fire event and year**, never by random patch.
- Train the spread model only on layers our own agents produce, so it can run on the real datacube.
- Keep the pipeline working in plain Python. Wrap it in Ray / LangGraph / SPADE only at the end, if required.

## Next steps

1. Phase 1: each owner replaces their agent's baseline with a trained model and reports metrics.
2. Phase 2–3 (done on Evros 2023): day-by-day sequence, CA calibration and Sentinel-2 ablation; results in [docs/case_study.md](docs/case_study.md). Next: same on Evia 2021, trained spread U-Net in the benchmark, learned ensemble weights.
3. Phase 3: benchmark persistence vs U-Net vs ConvLSTM vs CA on the same test fires.
4. Phase 4: FIRMS-triggered runs over several days (the coordinator already keeps `last_known` layers between runs).
5. Phase 5: dashboard with a Folium map, plus the report.
