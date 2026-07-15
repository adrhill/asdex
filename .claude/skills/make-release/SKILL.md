---
name: make-release
description: Prepare a new release by updating the changelog and version
disable-model-invocation: true
---

Prepare a new release for asdex.

## Context

- Current version: !`grep '^version' pyproject.toml`
- Latest git tag: !`git describe --tags --abbrev=0`
- Commits since last tag: (run `git log <latest-tag>..HEAD --oneline` using the tag above)

## Instructions

1. Check out `main` and pull latest: `git checkout main && git pull`.

2. Determine the version bump using [semver](https://semver.org/) (MAJOR.MINOR.PATCH).
   The highest bump among all commits wins.
   While the version has a leading 0, the public API is not stable,
   so shift bump levels down: breaking → MINOR, everything else → PATCH.
   After 1.0.0: breaking → MAJOR, `feat:` → MINOR, `fix:` → PATCH.

3. For commits that also fix bugs or add features beyond what the commit type suggests
   (check PR descriptions with `gh pr view`), include those as separate changelog entries.

4. Map commit types to changelog badge types:
   - `badge-breaking` ← any `!` (e.g. `feat!:`, `fix!:`)
   - `badge-feature` ← `feat:`
   - `badge-enhancement` ← `perf:`, or `refactor:` when the PR description shows a user-facing improvement
   - `badge-bugfix` ← `fix:`
   - `badge-maintenance` ← `refactor:`, `chore:` with a user-facing effect
   - `badge-docs` ← `docs:`
   - The changelog is for users. Only include entries with a real user-facing impact,
     even when the commit type maps to a badge (e.g. skip `test:` commits, tooling-only
     changes, and internal commits like `claude:`, `ci:`). When unsure, ask before including it.

5. Update `CHANGELOG.md`:
   - Add a new `` ## Version `vX.Y.Z` `` section above the previous release
   - Order entries by badge type: breaking, feature, enhancement, bugfix, maintenance, docs
   - Each entry links to its PR using the existing badge format.
     If a commit has no PR, link to the commit using the first 6 characters of the hash.
   - Add PR/commit link references in numerical order

6. Update `version` in `pyproject.toml`

7. Show the user a summary of changes.
   Wait for user confirmation before proceeding to commit.

8. Commit as `` asdex `vX.Y.Z` ``.
   IMPORTANT: Stage both `CHANGELOG.md` AND `pyproject.toml` in the commit.
   Ask the user if there are any other uncommitted files that should be included.

9. Push the commit, then tag with `git tag vX.Y.Z` and push the tag with `git push --tags`.
   `vX.Y.Z` is the new version number from `pyproject.toml`.

10. Create a GitHub release:
    - Fetch the merged pull requests for this release from GitHub's auto-generated notes.
      These are bounded by the tag range, unlike a `merged:<date>` search, which misattributes
      PRs merged on the previous tag's day:
      ```
      gh api repos/adrhill/asdex/releases/generate-notes \
        -f tag_name=vX.Y.Z -f previous_tag_name=<previous-tag> --jq .body \
        | sed -nE 's~^\* (.+) by (@[^ ]+) in https://github.com/[^/]+/[^/]+/pull/([0-9]+)~- \1 (#\3) (\2)~p'
      ```
      Each line becomes `- <title> (#N) (@author)`, listing every merged PR (not just the
      user-facing ones in the changelog).
    - Fetch closed issues since the previous tag:
      ```
      gh api "repos/adrhill/asdex/issues?state=closed&since=<prev-tag-date>" \
        --jq '.[] | select(.pull_request == null) | "- \(.title) (#\(.number))"'
      ```
      Get the previous tag date with: `git log -1 --format=%aI <previous-tag>`
    - The same `generate-notes` body carries a "New Contributors" section listing first-time
      contributors. Extract it if present.
    - Compose release notes in this order:
      1. Our badge-formatted changelog entries + link references
      2. "Merged pull requests" section, listing every merged PR with its author `@`-tagged
      3. "Closed issues" section (if any issues were closed)
      4. "New Contributors" section (if any, from GitHub's generated notes)
      5. Full Changelog link: `**Full Changelog**: https://github.com/adrhill/asdex/compare/<prev>...vX.Y.Z`
    - Tag each merged PR's author with a bare `@login` mention. GitHub reads these mentions and
      renders the release's "Contributors" avatar strip, so this credits everyone who contributed.
      Keep the "New Contributors" section alongside it: the strip does not distinguish first-time
      contributors, so that callout is not redundant.
    - Give every issue and PR number an explicit reference-style link. Do not rely on GitHub
      auto-linking a bare `#N`. Add a link definition for each in numerical order, pointing PR
      numbers at `.../pull/N` and issue numbers at `.../issues/N`.
    - Create the release: `gh release create vX.Y.Z --title "vX.Y.Z" --notes "..."`
