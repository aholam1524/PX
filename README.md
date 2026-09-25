# PX

App repo. The agent factory lives in [aholam1524/ASD](https://github.com/aholam1524/ASD) (`main`). Actions in this repo check that code out to `.asd-factory/` and run it against PX.

## Housing price analyzer

Interactive map and analysis for **Finnish housing prices** by postal-code area. The Streamlit app has **Map**, **Compare**, and **Relationships** tabs. On **Map**, a **MapLibre choropleth** colours postal-code boundaries by your chosen metric (price per square metre, nominal or inflation-adjusted 1- or 5-year change, or sales in the last four quarters). Use the sidebar to pick quarter and building type; hover for details, click or search to select an area. The **area detail panel** (beside the map) shows key figures, rank and reliability, a quarterly **trend chart** with municipality and national comparison lines (gaps where data is missing), sales volume from 2020, flagged unusual quarter-on-quarter moves, and a CSV download; you can **add the selected area to comparison**. On **Compare**, pick up to four areas (search by postal code or name) to see a side-by-side metrics table, one multi-series price chart (optional index to 100 at the start), and **similar areas** for a chosen comparison area. The **Relationships** tab scatter-plots price per m² against Paavo income and demographic shares (with correlation and an exclusion count). The map can colour areas by a rough **price-to-income** ratio. Grey areas have no published price; lighter borders and hover notes mark low-reliability estimates.

**Run locally** (Python 3.12):

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Run without network or snapshot** (small fixture data under `tests/fixtures/`):

```bash
HOUSING_USE_FIXTURES=1 streamlit run app/streamlit_app.py
```

### Hosting

Deploy on [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud) from this public GitHub repository (you connect the repo in the browser; no deploy step in this repo).

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and authorize access to this repository.
2. Click **New app**, choose **aholam1524/PX** (or your fork), then pick the branch:
   - **`dev`** — preview app that tracks the integration branch.
   - **`main`** — stable app after releases merge to `main`.
3. Set **Main file path** to `app/streamlit_app.py`.
4. Under **Advanced settings**, set **Python version** to **3.12**. No secrets or environment variables are required for the default snapshot-backed app.
5. Deploy. The app reads committed files under `data/snapshot/`; it does not download live Statistics Finland data on first load.

Two apps (one on `dev`, one on `main`) are a practical setup: preview changes on `dev` before the snapshot and code reach `main`. For limits, billing, and platform behavior, see Streamlit’s [Community Cloud documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud).

- **`housing_analyzer/`** — data, analysis, map helpers (`housing_analyzer/map.py`), and area detail panel logic (`housing_analyzer/panel.py`)
- **`app/streamlit_app.py`** — Streamlit UI
- **`tests/`** — pytest; fixtures under `tests/fixtures/` for offline runs

### Data

Quarterly housing-company prices and transaction counts by postal-code area come from Statistics Finland’s table [“Prices per square meter of old dwellings in housing companies and numbers of transactions by postal code area, quarterly”](https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/ashi/13mt.px) (PxWeb API). The loader lives in `housing_analyzer/data/prices.py` (`load_prices()` / `fetch_prices()`).

Postal-code area boundaries for the map come from Statistics Finland’s WFS service [`geo.stat.fi/geoserver/postialue/wfs`](https://geo.stat.fi/geoserver/postialue/wfs) (feature type `postialue:pno_2022`, matching the 2022-01-01 postal-code list used in the price table). The loader lives in `housing_analyzer/data/boundaries.py` (`load_boundaries()` / `join_prices_to_areas()`).

**Paavo demographics and income** (population, median disposable income, share aged 65+, share with higher university degree) come from Statistics Finland’s open database [Postinumeroalueittainen_avoin_tieto](https://pxdata.stat.fi/PxWeb/api/v1/en/Postinumeroalueittainen_avoin_tieto/) (`uusin` tables **12ey.px**, **12f1.px**, **12ez.px**, reference year **2024**, postal variable `postinumeroalue_4_20260101`). Loader: `housing_analyzer/data/demographics.py` (`load_demographics()` / `fetch_demographics()`). Paavo uses a newer postal-code edition than the 2022 price table and boundaries; codes that exist in only one dataset stay missing on join and are reported in logs, not imputed.

Parsed downloads are cached under `data/raw/` (git-ignored). The app prefers a committed snapshot under `data/snapshot/` (`prices.csv.gz`, `boundaries.geojson.gz`, `cpi.csv.gz`, `demographics.csv.gz`, `manifest.json`) so cold starts do not hit the live APIs.

**Real (inflation-adjusted) prices** express nominal euro per square metre in the purchasing power of a chosen CPI quarter (by default the latest complete quarter), using Statistics Finland’s overall consumer price index. Source: [Consumer Price Indices, overall index, monthly (`11xs`)](https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/khi/11xs.px), series `ip_0_2015` (2015=100).

### Refreshing the data

After this workflow is on **`main`**, open [Refresh housing data](https://github.com/aholam1524/PX/actions/workflows/refresh-data.yml) in GitHub Actions and click **Run workflow**. It fetches the latest prices and boundaries, rebuilds `data/snapshot/`, and opens a pull request into `dev` when anything changed (merge that PR to update the snapshot).

Quarterly is enough for housing statistics; you only need a refresh when Statistics Finland publishes new quarters or boundary editions. Each refresh adds a few megabytes to git history, so avoid running it more often than necessary.

### Analysis

Pure helpers in `housing_analyzer/analysis/` turn the tidy price table into map and panel metrics. They never call the network and never modify the input DataFrame.

- **Percentage change** (`pct_change`, `real_pct_change`): For each postal-code area and building type, compares the price in a quarter to the price **four calendar quarters earlier** (one year when `quarters=4`, five years when `quarters=20`). Real changes use `real_price_per_sqm` from `to_real()` and the quarterly CPI (`cpi_by_quarter`). If either price is missing, the change is missing—not zero.
- **Reliability** (`reliability`): Labels each row using transaction counts in a trailing window (default: sum of the last four quarters). `"ok"` means enough transactions for a stable average; `"low"` means fewer; `"none"` means the price itself is missing; `"unknown"` applies to quarters before 2020, when public transaction counts are not available.
- **Rank and percentile** (`rank_percentile`): For one quarter, ranks areas by price (1 = most expensive). Areas without a price are not ranked. Tied prices share the same rank. When no single building type is chosen, the area price is a **transaction-weighted** average across types that have transaction counts; if none do, a simple mean of available prices is used—the result includes which method was applied.
- **Regional average** (`regional_average`): For a caller-supplied mapping from postal codes to groups (for example municipalities), computes a group average for one quarter. Uses transaction-weighted averaging across areas when weights exist; otherwise falls back to a simple mean, and records which method was used.
- **Area summary** (`summarize_area`): One dict with price, one- and five-year changes, reliability, rank, and percentile for a selected area and quarter.
- **All areas at once** (`summarize_areas`, `area_prices_at`, `reliability_at`): The same numbers for every postal-code area in one pass over the data. The map uses these instead of calling `summarize_area` per area, so drawing the map does not slow down as the number of areas grows. A test checks that both give identical results.
- **Compare tab** (`housing_analyzer/analysis/compare.py`): Builds the comparison table and multi-area price chart from the same summaries and trailing sales as the map.
- **Price-to-income ratio** (`price_to_income_ratio` in `housing_analyzer/analysis/relationships.py`): Latest nominal EUR/m² divided by the area’s **median disposable monetary income (EUR/year)** from Paavo. This is a coarse map overlay only—it ignores dwelling size, household composition, and loan costs.
- **Relationships tab** (`housing_analyzer/analysis/relationships.py`): Scatter plots of price per m² vs income and demographic shares for areas with OK price reliability and complete Paavo fields; shows sample size, Pearson *r*, and a short note that correlation is not causation.
- **Similar areas** (`housing_analyzer/analysis/similar_areas.py`): Among areas with reliability `"ok"` and complete feature data, ranks the five closest to a target using weighted Euclidean distance on **z-scored** price per m², 5-year change, and **median income when available for the target** (`SIMILARITY_FEATURES` / `OPTIONAL_SIMILARITY_FEATURES`). Ties break on postal code.

Nothing in this repository is investment advice.

Describe the work in Cursor chat in this repo. The agent files a GitHub issue; that **queues** the work (label `factory-queued`). You run **Start factory** in Actions to begin Dev on the oldest queued issue. Code moves `feature/N-slug` → `dev` → `test` → `main`. After review, the feature PR merges into `dev` automatically when Factory CI is green; you merge into `main`. Merge into `test` is automatic when the Test agent reports PASS and CI is green.

When Dev pushes `feature/*`, the factory opens a PR **into `dev`**, adds `agent-review`, and the **Claude review** workflow runs in GitHub Actions (even if the Dev agent forgot to open the PR).

```text
You describe work in Cursor
    → Agent files a GitHub issue (queued)
    → Start factory (Actions) → Dev agent: branch feature/<issue>-<slug>
    → Push opens PR into dev, labels agent-review
    → Claude review in Actions → one Claude fix pass (Sonnet 5) → automatic merge into dev if Factory CI is green
    → Factory opens PR dev → test and starts Test (Test runs only on this promotion)
    → Test PASS + CI green → automatic merge into test
    → Factory opens PR test → main (Test does not run again on this PR)
    → You merge into main
```

Shared factory behavior (status labels, Fixer, Conflict, one open feature PR at a time, retry labels) is documented in the [ASD README](https://github.com/aholam1524/ASD#readme). In PX: **Fixer** runs when Test fails on the `dev` → `test` PR and retries on the feature branch; **Conflict** runs when that promotion PR has merge conflicts; only **one** feature PR into `dev` blocks the next Dev launch until it merges. To retry a failed step, add `agent-dev` on an **issue** or `agent-test`, `agent-review`, `agent-fix`, or `agent-conflict` on a **PR** (the dispatcher creates factory status labels as needed).

## One-time setup

1. Repo secret `CURSOR_API_KEY` from [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations).

2. Repo secret **`CLAUDE_CODE_OAUTH_TOKEN`**: a Claude subscription token (Pro, Max, Team, or Enterprise — not an API key). On your machine run `claude setup-token`, then add the token as a repository secret with this name. Review and the post-review fix pass run through [claude-code-action](https://github.com/anthropics/claude-code-action) on **Claude Sonnet 5** and **count against your Claude subscription usage limits**. The review job uses `${{ secrets.GITHUB_TOKEN }}` for PR comments; the fix pass pushes with `FACTORY_GITHUB_TOKEN`. You do not need the [Claude GitHub App](https://github.com/apps/claude) unless you prefer app-based auth instead.

3. Grant this repository to the Cursor GitHub app (clone + open PRs).

4. Add repo secret **`FACTORY_GITHUB_TOKEN`**: a GitHub classic PAT (scope `repo`) or a fine-grained token with Contents, Issues, and Pull requests read/write on **this repo and ASD**, created as **your user**. The factory uses it to open PRs as you and to check out ASD. PRs opened as `github-actions[bot]` sit on **Approve workflows** and follow-up jobs (including `test` → `main`) may never start. Without this secret those waits come back.

5. Create long-lived branches if they do not exist (the factory will also create them from `main` on first promote):

   ```bash
   git fetch origin
   git checkout main
   git pull
   git checkout -b dev && git push -u origin dev
   git checkout main
   git checkout -b test && git push -u origin test
   ```

If `test` is branch-protected, allow GitHub Actions to merge or auto-merge into `test` will fail.

In the repo: Settings → Actions → General → Workflow permissions → **Read and write**. Otherwise the factory cannot create `dev`/`test` or open promotion PRs.

You do not approve workflow runs. You only merge PRs into `main` after Test on the `dev` → `test` promotion PR.

## Factory version

Repository variable **`ASD_FACTORY_REF`** selects which revision of [aholam1524/ASD](https://github.com/aholam1524/ASD) the workflows check out into `.asd-factory/`. It can be a tag, branch name, or full commit SHA. Set it under **Settings → Secrets and variables → Actions → Variables**. If it is unset, workflows use ASD `main`.

After an ASD change lands on `main` and you have tried it in PX, set `ASD_FACTORY_REF` to that commit SHA to pin the factory until you deliberately change the variable to upgrade.

Each factory workflow run logs the resolved ref and the checked-out commit (`git -C .asd-factory rev-parse HEAD`) right after the ASD checkout.

## How to start

1. Open [Start factory](https://github.com/aholam1524/PX/actions/workflows/start-factory.yml) in GitHub Actions.
2. Click **Run workflow** (no inputs). The branch dropdown does not matter. Start factory always checks out this repo's `dev` and factory code from ASD at **`ASD_FACTORY_REF`** or `main` when unset.

That starts Dev on the oldest open issue with the `factory-queued` label. If the queue is empty, the run succeeds and does nothing. If a `feature/*` → `dev` PR is already open, Dev is not started until that PR is merged (or add `agent-dev` on a specific issue to retry that issue only).

## How to use it

1. In Cursor, say what you want built. The agent creates the issue in this repo; it is queued automatically. No label required.
2. Run **Start factory** (see above). Wait for a PR from `feature/<number>-<slug>` **into `dev`** (opened on push if Dev only pushed a branch). Adding `agent-review` triggers **Claude review** in Actions, then one **Claude fix pass** (Sonnet 5), then an automatic merge into `dev` when Factory CI is green on the PR head.
3. The factory opens `dev` → `test`, runs the **Test** cloud agent on that promotion PR only, then Factory CI. On PASS + green CI it merges into `test` and opens `test` → `main`.
4. Read Test comments on the `dev` → `test` PR. You merge into `main`. That closes the ticket issue (label `factory-done`). After that merge, the next open issue with `factory-queued` starts Dev automatically (same rules as **Start factory**).

Happy path needs no manual labels. `main` is never auto-merged.

Watch SDK-launched agents (Dev, Test, Fixer, Conflict) in Cursor: Agents → Filter → Source → SDK. Review, the fix pass, and the merge-into-`dev` step run in the **Claude review** workflow, not as Cursor cloud agents.

### Claude fix pass

After review, one **fix pass** may commit and push to the feature branch. It must not edit `.cursor/` or anything under `.asd-factory/`, and it must not create new files under `.github/`. It **may** edit a file under `.github/` only when that file is already part of the PR diff against `dev` (typical for factory wiring changes).

Each fix pass posts a **Fix pass result** PR comment: a table with one row per review finding (`fixed` with commit details, or `not fixed` with a reason), plus an HTML marker `<!-- factory:fix-result:all-addressed -->` or `<!-- factory:fix-result:open-findings -->`. The workflow also posts whether any commits were pushed (`Fix pass pushed N commit(s): …` or `Fix pass pushed no commits`). The fix-once marker is posted only after a **successful** fix pass; a failed fix pass leaves no marker so re-adding `agent-review` retries the fix.

Repository variable **`FACTORY_BLOCK_ON_OPEN_FINDINGS`**: set to `true` under **Settings → Secrets and variables → Actions → Variables** to skip automatic merge into `dev` when the fix pass ends with `open-findings`. When unset or not `true` (default), merge proceeds as today and a one-line warning is added if findings were still open.

## Usage reporting

When **Claude review** runs on a feature PR (`agent-review`), the workflow posts **one** PR comment (updated after each job) with a table of **review**, **fix**, and **merge** job usage: Claude turns, input/output/cache token counts, an **API-equivalent cost estimate**, Linux runner minutes (billed, rounded up), and estimated Linux cost. The same table is written to each job’s GitHub Actions step summary. If any of those jobs fails, it also posts **one** comment per workflow run (with a run id marker) listing the failed jobs and linking to the Actions run, and sets the linked issue (`Closes #N` in the PR body) to `factory-blocked` unless it is already `factory-done`. Re-add `agent-review` to retry.

Token cost in that comment is **not** what you pay: review and fix use your Claude subscription (`CLAUDE_CODE_OAUTH_TOKEN`), so the dollar figure is only an API-equivalent estimate. On this **public** repo, hosted Linux runner minutes are **free** (`billed: $0.00`); the comment still shows what Actions would cost on a private repo. Override the per-minute rate with the repository variable **`ACTIONS_LINUX_RATE_PER_MIN`** (default `0.006`).

**Cursor cloud agents** (Dev, Test, Fixer, Conflict) are **not** included in that comment; their usage is on the [Cursor dashboard](https://cursor.com/dashboard).
