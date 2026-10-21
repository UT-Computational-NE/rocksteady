# AGENTS.md

## Agent skills

### Issue tracker

Issues live in GitHub Issues for UT-Computational-NE/rocksteady (via `gh`). Claim a ticket before starting it. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/` (created lazily). See `docs/agents/domain.md`.

### Worktrees and branches

Use one worktree per ticket: `git worktree add .claude/worktrees/feat-<n> -b feat/<n>-<slug> origin/main`. A stacked PR branches from its blocker's branch instead. The environment's editable install may point at another checkout, so run code with `tools/check.sh` or with `PYTHONPATH=src`.

### Checks

`tools/check.sh` runs CI's fast checks: the tests without `slow`, and the strict docs build. `tests/test_vocabulary.py` fails on phrases that `GLOSSARY.md` retires. When a phrase is always wrong, add it there.

### Reviews and merging

`npl436-bot` reviews PRs. Merge only when the user asks. See `docs/agents/review.md`.
