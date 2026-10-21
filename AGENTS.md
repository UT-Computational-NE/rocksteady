# AGENTS.md

## Agent skills

### Issue tracker

Issues live in GitHub Issues for UT-Computational-NE/rocksteady (via `gh`). Claim a ticket before starting it. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/` (created lazily). See `docs/agents/domain.md`.

### Worktrees and branches

Use one worktree per ticket: `git worktree add .claude/worktrees/feat-<n> -b feat/<n>-<slug> origin/main`. A stacked PR branches from its blocker's branch instead. The environment's editable install may point at another checkout, so run code with `tools/check.sh` or with `PYTHONPATH=src`, and install packages by name: `pip install -e .` in a worktree points that shared install at the worktree.

### Checks

`tools/check.sh` runs CI's fast checks: the tests without `slow`, mypy, and the strict docs build. `tools/check.sh <pytest args>` runs only that selection, without mypy or the docs build, and needs only the test extra. `tests/test_vocabulary.py` fails on a fixed list of phrases (`AVOIDED`) that `GLOSSARY.md` avoids and that are always wrong. When you find another such phrase, add it to `AVOIDED` with a near-miss: correct text close to it that the scan must not flag.

### Reviews and merging

`npl436-bot` reviews PRs. Merge only when the user asks. See `docs/agents/review.md`.
