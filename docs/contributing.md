# Contributor's Guide and AI Policy

This document describes the practices we follow when contributing to asdex.
Making them explicit helps new contributors understand what is expected of them.
They aim to make what can otherwise feel like a mysterious process clear to anyone opening their first issue or pull request (PR).

## Community Standards

Interactions with people in the community must always follow the [Python Community Code of Conduct](https://policies.python.org/python.org/code-of-conduct/),
including in pull requests, issues, and discussions.

## Contributing PRs

* [Open an issue](https://github.com/adrhill/asdex/issues) before starting work, and declare in it that you intend to contribute a fix.
    We are very happy to help discuss how to tackle it.
    Agreeing on a plan before implementing is less effort for both PR authors and reviewers,
    and maintainers can point you to the files a change belongs in.
* A PR should affect as little code as possible.
    Reviewing a couple of small PRs is faster than reviewing one large PR.
* PRs should match the existing code style present in the file.
    * Lint, format, and type checks are enforced by the git hooks from the [Development Setup](#development-setup) below.
    * Underscore-prefix any name that is not part of the public API.
* PRs affecting the public API, including adding new features, must update the public documentation.
* Comments and docstrings should make the code accessible.
* PRs that change code must have appropriate tests.
* Changes to the code must be made via PR, not pushing to `main`.

## AI Policy

asdex is written with assistance of LLM coding agents (colloquially referred to as AI),
building on existing, hand-written, pre-AI code by the same authors.[^prior-work]
This section describes how we use AI and how we expect contributors to use it.

[^prior-work]: [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl), [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl), and [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl).

* All asdex code is read and reviewed multiple times before it is merged.
    Code changes are iterated over several times, as the commit history of most PRs shows.
    **We do not permit vibe-coding**, by which we mean unreviewed code that its human author did not verify.
* **We welcome AI-written and vibe-coded bug reports**. Do not shy away from using AI to file them, even if you are uncertain, as they are easy for us to verify.
* **We welcome AI-assisted PRs, but only under a few conditions**:
    1. You must first [open an issue](https://github.com/adrhill/asdex/issues) and get a maintainer's approval.
        We will generally say yes and are glad to help.
        Issues allow us to scope the work so nobody ends up having to review thousands of lines at once.
    2. You must understand your own code, be able to walk us through it, and talk it over during the PR review.
    3. You are not allowed to use AI to write your answers in PR reviews (with the exception of spell-checking and translation).

Our reasoning is that we want to support (junior) developers and keep growing the open source community.
We gladly invest the extra review effort when a human learns from it, but an AI takes nothing away from our feedback.
If a contributor only forwards prompts between us and an AI, we can develop asdex faster by prompting and reviewing our own code.
In that case, please open a feature request or bug report instead.

## Commit Messages

Commit messages should follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification:

* Prefix each message with a type such as `feat:`, `fix:`, `docs:`, `refactor:`, or `test:`.
* For breaking changes, append `!` after the type, for example `feat!:`.
    We define what we consider a breaking change [here](#changes-that-are-considered-breaking).
    If you are uncertain, err on the conservative side and prefix a commit as breaking, then ask a maintainer for advice.

## Reviewing, Approving, and Merging PRs

* PRs should pass CI tests before being merged.
* PRs must have approval from someone who has merge rights.

## Releases

* A release should be made as soon as possible after a bugfix PR is merged.
* Care and consideration should be given as to when to make a breaking release.
* Unreleased changes accumulate in [`CHANGELOG.md`](https://github.com/adrhill/asdex/blob/main/CHANGELOG.md), and a release adds a matching changelog entry and bumps the `version` field in [`pyproject.toml`](https://github.com/adrhill/asdex/blob/main/pyproject.toml).
* A maintainer cuts the release by tagging the release commit `vX.Y.Z` and publishing a GitHub release.

## Becoming a maintainer (gaining merge rights)

* Before becoming a maintainer, it is usual to:
    * contribute several PRs,
    * review constructively and kindly several PRs,
    * contribute meaningfully to several discussions on issues.
* Maintainers are listed in the `maintainers` field of [`pyproject.toml`](https://github.com/adrhill/asdex/blob/main/pyproject.toml). When someone is added as a maintainer, they should open a PR adding their name and contact there.
* You may ask to be added as a maintainer. It is not rude to ask.

---

## Development Setup

### Installing dependencies

asdex uses [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.

First, install uv by following the [official instructions](https://docs.astral.sh/uv/getting-started/installation/),
for example with the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the development dependencies.
`uv sync` creates a virtual environment and installs asdex together with the `dev` dependency group:

```bash
uv sync
```

### Git hooks

Pre-commit hooks are checks that run automatically on your staged files each time you create a commit,
stopping the commit if any of them fail.
asdex uses [prek](https://prek.j178.dev/) to run its lint, format, and type-check hooks,
configured in [`.pre-commit-config.yaml`](https://github.com/adrhill/asdex/blob/main/.pre-commit-config.yaml).
`prek` is a drop-in reimplementation of `pre-commit` and ships with the `dev` group.
Install the hooks once so they run automatically on each commit:

```bash
uv run prek install
```

Once installed, the hooks run on every commit without any further action.
To manually check the entire repository at once instead of only the staged files, run `uv run prek run --all-files`.

### Running Tests

Run the test suite with pytest:

```bash
uv run pytest                          # the whole suite
uv run pytest tests/test_coloring.py   # a single file
```

By default the suite skips the `slow`, `benchmark`, and `cutest` tests, which only run in CI.
Pass `-m` with a marker to run a subset:

```bash
uv run pytest -m jacobian      # only sparse Jacobian tests
uv run pytest -m "not slow"    # skip slow tests
```

The default selection and the full list of markers are defined in the `[tool.pytest.ini_options]` table of
[`pyproject.toml`](https://github.com/adrhill/asdex/blob/main/pyproject.toml),
and the markers are registered in
[`tests/conftest.py`](https://github.com/adrhill/asdex/blob/main/tests/conftest.py).

---

## Further Guidance

### Guidance on contributing PRs

* You should usually open an issue about a bug or possible improvement before opening a PR with a solution.
* PRs should do a single thing, so that they are easier to review.
    * For example, fix one bug, or update compatibility, rather than fixing a bunch of bugs, and updating compatibility, and adding a new feature.
* PRs should add tests which cover the new or fixed functionality.
* PRs that move code should not also change code, so that they are easier to review.
    * If only moving code, review for correctness is not required.
    * If only changing code, then the diff makes it clear what lines have changed.
* PRs with large improvements to style should not also change functionality.
    * This is to avoid making large diffs that are not the focus of the PR.
    * While it is often helpful to fix a few typos in comments on the way past, it is different from using a regex or formatter on the whole project to fix spacing around operators.
* PRs introducing breaking changes should make this clear when opening the PR.
* You should not push commits with commented-out tests.
    * If pushing a commit for which a test is expected to fail, mark it with `@pytest.mark.xfail`.
    * Commenting out tests while developing locally is okay, but committing a commented-out test increases the risk of it silently not being run when it should be.
* You should not squash down commits while review is still ongoing.
    * Squashing commits prevents the reviewer from seeing what commits have been added since the last review.
* You should help __review__ your PRs, even though you cannot __approve__ your own PRs.
    * For instance, start the review process by commenting on why certain bits of the code changed, or highlighting places where you would particularly like reviewer feedback.

### Guidance on reviewing PRs

* Review comments should be phrased as questions, as it shows you are open to new ideas.
    * For instance, “Why did you change this to X? Doesn’t that prevent Y?” rather than “You should not have changed this, it will prevent Y”.
* Small review suggestions, such as typo fixes, should make use of the `suggested change` feature.
    * This makes it easier and more likely for all the smaller changes to be made.
* Reviewers should continue acting as reviewers until the PR is merged.

### Guidance on Package Releases

#### Incrementing the package version

* Follow [Semantic Versioning 2.0](https://semver.org).
    The version lives in the `version` field of [`pyproject.toml`](https://github.com/adrhill/asdex/blob/main/pyproject.toml).
* The highest bump implied by the commits since the last release wins.
    Map the [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) type of each commit to a bump:
    * From 1.0.0 onwards: a breaking change (`!`) bumps MAJOR, `feat:` bumps MINOR, and `fix:` bumps PATCH.
    * While the version has a leading `0`, the public API is not stable, so the bump levels shift down: a breaking change bumps MINOR, and everything else bumps PATCH.
* Introducing deprecations is not breaking, but removing deprecations is breaking.
* Breaking releases have a cost, since downstream users must adapt their code, so there should be a clear benefit before making one.

#### Preparing a release

Unreleased changes accumulate in [`CHANGELOG.md`](https://github.com/adrhill/asdex/blob/main/CHANGELOG.md).
Between releases, the `version` field in [`pyproject.toml`](https://github.com/adrhill/asdex/blob/main/pyproject.toml) may carry a `-DEV` suffix
(for example `0.5.1-DEV`) to signal that the checkout is ahead of the last tagged release.
This is encouraged but not strictly enforced, and cutting a release drops the suffix so that the `version` field names the release exactly.

!!! tip "Automated releases"

    Maintainers with access to [Claude Code](https://claude.com/claude-code) can run the [`/make-release`](https://github.com/adrhill/asdex/blob/main/.claude/skills/make-release/SKILL.md) skill,
    which walks through every step below (changelog entry, version bump, tag, and GitHub release).

A maintainer prepares a release as follows:

* Determine the new version from the commits since the last tag, following the bump rules above.
* Add a `` ## Version `vX.Y.Z` `` section to [`CHANGELOG.md`](https://github.com/adrhill/asdex/blob/main/CHANGELOG.md), ordering entries by badge type (breaking, feature, enhancement, bugfix, maintenance, docs) and linking each entry to its PR.
* Bump the `version` field in [`pyproject.toml`](https://github.com/adrhill/asdex/blob/main/pyproject.toml) to the new version.
* Commit [`CHANGELOG.md`](https://github.com/adrhill/asdex/blob/main/CHANGELOG.md) and [`pyproject.toml`](https://github.com/adrhill/asdex/blob/main/pyproject.toml) together, tag the commit `vX.Y.Z`, and push the tag.
* Publish a GitHub release for the tag, including the changelog entries and any issues closed since the previous release.

#### Changing dependency compatibility

* Generally, changing dependency compatibility should be a non-breaking feature.
    * i.e. pre-1.0, change the patch version number; post-1.0, change the minor version number.
    * For instance, adding or removing compatibility with a particular __version__ of a current dependency, which may or may not require internal code changes.
    * This also applies when adding or removing packages as dependencies.
    * The new feature in question is the ability to use with a different set of packages.
* Changing a dependency to resolve a bug is a bug-fix.
    * i.e. pre/post-1.0 change patch version number.
* Changing compatibility with dependencies **may** be a breaking release, if it breaks the user-facing interface.
    That is to say, if the dependency’s API leaks into your API.
    There are three ways that this can happen:
    * Reexporting a function that has changed.
    * Returning an object of a type whose behavior has changed.
    * Subclassing an object that has changed.

---

## Definition of Public API

The public API consists of the names re-exported from the top-level `asdex` module,
that is, the names listed in [`asdex.__all__`](https://github.com/adrhill/asdex/blob/main/src/asdex/__init__.py) and reachable as `asdex.<name>`.
These are the only names covered by the compatibility guarantees below,
and they should always be imported from the main module, for example `from asdex import jacobian`.

Everything else is internal and may change at any point without notice:

* Any module, function, class, or attribute whose name is prefixed with an underscore,
    such as `asdex._pattern` or `asdex._arguments`.
* The internal file and module layout under [`src/asdex/`](https://github.com/adrhill/asdex/tree/main/src/asdex).
    Import paths such as `asdex.coloring`, `asdex.detection`, and `asdex.decompression` are implementation details,
    even when they expose names that are themselves part of the public API.
    Import those names from `asdex` directly instead.

Do not rely on underscored names or internal file paths.
If something you need is not reachable from the top-level `asdex` module, please open an issue.

### Changes that are considered breaking

* Breaking changes are changes which break functionality in the public API, as defined above.
* Removing or renaming a public name, or changing the signature or documented behavior of a public function, is breaking.
* Changes which break a documentation example or tutorial are breaking.

### Changes that are not considered breaking

Everything on this list can, in theory, break users' code. See [XKCD#1172](https://xkcd.com/1172/).
However, we consider changes to these things to be non-breaking from the perspective of package versioning.

* **Bugs:** We may make backwards incompatible behavior changes if the current implementation is clearly broken, that is, if it contradicts the documentation or if a well-understood behavior is not properly implemented due to a bug.
* **Internal changes:** Non-public API may be changed or removed.
* **Exception behavior:**
    * Exceptions may be replaced with non-error behavior. For instance, we may change a function to compute a result instead of raising an exception, even if that error is documented.
    * Error message text may change.
    * Exception types may change unless the exception type for a specific error condition is specified in the documentation.
* **Floating-point numerical details:** The specific floating-point values may change at any time.
    Users should rely only on approximate accuracy, numerical stability, or statistical properties, not on the specific bits computed.
* **New public names**: Adding a new name to the public API is never considered breaking.
    However, one should consider carefully before adding a commonly used name that might clash with an existing one.
* **New base classes and broader types**:
    * A new base class may be added to an existing public class, as long as its documented behavior is preserved.
    * A concrete return type may be replaced by a more general type that still supports every documented use of the original.
* **Changes to the string representation:** The output of `str` or `repr` on an object may change at any time.
    Users should not depend on the exact text, but rather on the meaning of the text.
    Changing the string representation often breaks downstream tests, because it is hard to write test cases that depend only on meaning.

(This guidance on non-breaking changes is inspired by [https://www.tensorflow.org/guide/versions](https://www.tensorflow.org/guide/versions).)

---

## Acknowledgements

These contribution guidelines have been adapted from the [SciML ColPrac](https://github.com/SciML/ColPrac) guidelines.
