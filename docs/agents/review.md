# Reviews and merging

- **Accounts.** PRs are opened as `npl436`, which cannot approve its own PRs. `npl436-bot` posts reviews. It has write access, so its approval satisfies the "Protect Main" ruleset, which requires one approving review and allows no bypass.
  - **Posting as the bot.** Pass the bot's token to just that one command: `GH_TOKEN=$(gh auth token --user npl436-bot) gh api --method POST repos/{owner}/{repo}/pulls/<n>/reviews --input review.json`. Pass the token only this way, through `GH_TOKEN=$(…)` on the same command line. Never echo, log or store it, and never put it in a file, a message or a commit.
  - **Keep the active account.** Every session on this machine shares the active `gh` account, so `gh auth switch` would make all of them act as the bot.
  - **Checking who posted:** add `--jq .user.login` to the API call; it prints `npl436-bot` when the bot posted.
- **Requesting a review, or requesting it again:** `gh api --method POST repos/{owner}/{repo}/pulls/<n>/requested_reviewers -f "reviewers[]=npl436-bot"`.
- **Merging.** Merge only when the user asks, with a merge commit, not a squash: `gh pr merge <n> --merge --subject "Merge #<n>: <title>"`.
- **Stacked PRs.** Before deleting a merged parent's branch, retarget the child to `main` with `gh pr edit <child> --base main`. Deleting the parent's branch first closes the child.
- **After a merge,** first retarget any stacked child PR (see above). Then remove the worktree (`git worktree remove`) and delete the branch locally and on the remote (`git branch -d`, `git push origin --delete`).
