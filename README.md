# PX

App repo. The agent factory lives in [aholam1524/ASD](https://github.com/aholam1524/ASD) (`main`). Actions in this repo check that code out to `.asd-factory/` and run it against PX.

Describe the work in Cursor chat in this repo. The agent files a GitHub issue; that **queues** the work (label `factory-queued`). You run **Start factory** in Actions to begin Dev on the oldest queued issue. Code moves `feature/N-slug` → `dev` → `test` → `main`. After Review, the factory merges the feature PR into `dev` automatically when CI is green; you merge into `main`. Merge into `test` is automatic when the Test agent reports PASS and CI is green.

When Dev pushes `feature/*`, the factory opens a PR **into `dev`**, adds `agent-review`, and the **Claude review** workflow runs in GitHub Actions (even if the Dev agent forgot to open the PR).

```text
You describe work in Cursor
    → Agent files a GitHub issue (queued)
    → Start factory (Actions) → Dev agent: branch feature/<issue>-<slug>
    → Push opens PR into dev, labels agent-review
    → Claude review in Actions → one Claude fix pass (Sonnet 5) → automatic merge into dev if CI is green
    → Factory opens PR dev → test and starts Test
    → Test PASS + CI green → automatic merge into test
    → Factory opens PR test → main (Test does not run again on this PR)
    → You merge into main
```

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

6. Retry labels (create once; ignore "already exists"):

   ```bash
   gh label create agent-dev --color 1D76DB --description "Retry the Dev cloud agent"
   gh label create agent-test --color 0E8A16 --description "Retry the Test cloud agent"
   gh label create agent-review --color 5319E7 --description "Retry Claude review in GitHub Actions"
   gh label create agent-fix --color D93F0B --description "Retry the Fixer cloud agent"
   gh label create agent-conflict --color FBCA04 --description "Retry the Conflict cloud agent"
   ```

   Factory status labels (the dispatcher also creates these if missing; ignore "already exists"):

   ```bash
   gh label create factory-queued --color C5DEF5 --description "Ticket queued for Dev"
   gh label create factory-dev --color 1D76DB --description "Dev agent running"
   gh label create factory-review --color 5319E7 --description "Review agent on feature PR"
   gh label create factory-waiting-dev --color BFDADC --description "Review/fix running; feature PR merges into dev when CI is green"
   gh label create factory-test --color 0E8A16 --description "Test agent on promotion PR"
   gh label create factory-fixer --color D93F0B --description "Fixer agent on feature PR"
   gh label create factory-conflict --color FBCA04 --description "Conflict agent on dev→test PR"
   gh label create factory-waiting-main --color FEF2C0 --description "dev→test promotion done; merge test→main PR to main"
   gh label create factory-done --color 006B75 --description "Work merged to main"
   gh label create factory-blocked --color B60205 --description "Test failed after Fixer; manual retry"
   ```

   | Label | Meaning |
   | --- | --- |
   | `factory-queued` | Issue filed; waiting for **Start factory** |
   | `factory-dev` | Dev agent running |
   | `factory-review` | Claude review running on the feature PR (Actions) |
   | `factory-waiting-dev` | Claude review and fix pass on the feature PR; automatic merge into `dev` when CI is green |
   | `factory-test` | Test agent on a promotion PR (`dev`→`test` or re-test after fix) |
   | `factory-fixer` | Fixer agent on the feature PR |
   | `factory-conflict` | Conflict agent on the `dev`→`test` PR |
   | `factory-waiting-main` | `dev`→`test` promotion complete; you merge `test`→`main` to `main` |
   | `factory-done` | Work merged to `main` |
   | `factory-blocked` | Test failed again after Fixer, or Claude review/fix/merge failed; retry with `agent-fix` / `agent-test` or `agent-review` on the PR |

   Each ticket keeps **one** `factory-*` status label at a time (retry labels stay separate).

If `test` is branch-protected, allow GitHub Actions to merge or auto-merge into `test` will fail.

In the repo: Settings → Actions → General → Workflow permissions → **Read and write**. Otherwise the factory cannot create `dev`/`test` or open promotion PRs.

You do not approve workflow runs. You only merge PRs into `main` after Test on the `dev` → `test` promotion PR.

## Factory version

Repository variable **`ASD_FACTORY_REF`** selects which revision of [aholam1524/ASD](https://github.com/aholam1524/ASD) the workflows check out into `.asd-factory/`. It can be a tag, branch name, or full commit SHA. Set it under **Settings → Secrets and variables → Actions → Variables**. If it is unset, workflows use ASD `main` (same as before).

After an ASD change lands on `main` and you have tried it in PX, set `ASD_FACTORY_REF` to that commit SHA to pin the factory until you deliberately change the variable to upgrade.

Each factory workflow run logs the resolved ref and the checked-out commit (`git -C .asd-factory rev-parse HEAD`) right after the ASD checkout.

## How to start

1. Open [Start factory](https://github.com/aholam1524/PX/actions/workflows/start-factory.yml) in GitHub Actions.
2. Click **Run workflow** (no inputs). The branch dropdown does not matter. Start factory always checks out this repo's `dev` and factory code from ASD `main`.

That starts Dev on the oldest open issue with the `factory-queued` label. If the queue is empty, the run succeeds and does nothing. If a `feature/*` → `dev` PR is already open, Dev is not started until that PR is merged (or add `agent-dev` on a specific issue to retry that issue only).

## How to use it

1. In Cursor, say what you want built. The agent creates the issue in this repo; it is queued automatically. No label required.
2. Run **Start factory** (see above). Wait for a PR from `feature/<number>-<slug>` **into `dev`** (opened on push if Dev only pushed a branch). Adding `agent-review` triggers **Claude review** in Actions, then one **Claude fix pass** (Sonnet 5), then an automatic merge into `dev` when CI is green.
3. The factory opens `dev` → `test`, runs Test, then CI. On PASS + green CI it merges into `test` and opens `test` → `main`.
4. Read Test comments on the `dev` → `test` PR. You merge into `main`. That closes the ticket issue (label `factory-done`). After that merge, the next open issue with `factory-queued` starts Dev automatically (same rules as **Start factory**).

Happy path needs no labels. To retry a failed launch: `agent-dev` on an **issue**; `agent-test`, `agent-review`, `agent-fix`, or `agent-conflict` on a **PR**.

Factory behavior (Fixer, Conflict, one open feature PR) is the ASD dispatcher. Feature PRs merge into `dev` automatically after review and fix; you merge into `main` yourself. `main` is never auto-merged.

Watch SDK-launched agents (Dev, Test, Fixer, Conflict) in Cursor: Agents → Filter → Source → SDK. Review and the fix pass run in the **Claude review** workflow, not as a Cursor cloud agent.

## Usage reporting

When **Claude review** runs on a feature PR (`agent-review`), the workflow posts **one** PR comment (updated after each job) with a table of **review**, **fix**, and **merge** job usage: Claude turns, input/output/cache token counts, an **API-equivalent cost estimate**, Linux runner minutes (billed, rounded up), and estimated Linux cost. The same table is written to each job’s GitHub Actions step summary. If any of those jobs fails, it also posts **one** comment per workflow run (with a run id marker) listing the failed jobs and linking to the Actions run, and sets the linked issue (`Closes #N` in the PR body) to `factory-blocked` unless it is already `factory-done`. Re-add `agent-review` to retry.

Token cost in that comment is **not** what you pay: review and fix use your Claude subscription (`CLAUDE_CODE_OAUTH_TOKEN`), so the dollar figure is only an API-equivalent estimate. On this **public** repo, hosted Linux runner minutes are **free** (`billed: $0.00`); the comment still shows what Actions would cost on a private repo. Override the per-minute rate with the repository variable `ACTIONS_LINUX_RATE_PER_MIN` (default `0.006`).

**Cursor cloud agents** (Dev, Test, Fixer, Conflict) are **not** included in that comment; their usage is on the [Cursor dashboard](https://cursor.com/dashboard).
