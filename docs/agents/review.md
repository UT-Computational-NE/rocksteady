# Reviews and merging

- **Accounts.** PRs are opened as `npl436`, which cannot approve its own PRs. `npl436-bot` posts reviews. It has write access, so its approval satisfies the "Protect Main" ruleset, which requires one approving review and allows no bypass.
- **Requesting a review, or requesting it again:** `gh api --method POST repos/UT-Computational-NE/rocksteady/pulls/<n>/requested_reviewers -f "reviewers[]=npl436-bot"`.
- **Merging.** Merge only when the user asks, with a merge commit, not a squash: `gh pr merge <n> --merge --subject "Merge #<n>: <title>"`.
- **Stacked PRs.** Before deleting a merged parent's branch, retarget the child to `main` with `gh pr edit <child> --base main`. Deleting the parent's branch first closes the child.
- **After a merge,** remove the worktree (`git worktree remove`) and delete the branch locally and on the remote (`git branch -d`, `git push origin --delete`).
