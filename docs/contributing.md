# ColPrac: Contributor's Guide on Collaborative Practices for Community Packages

[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/ColPrac/stable/)

This document describes some best practices for collaborating on repositories.
Following these practices makes it easier for contributors (new and old) to understand what is expected of them.
It should be linked to in the README.md.

There are many good practices that this document does not cover.
These include other members of the wider community reviewing pull requests (PRs) they are interested in, and maintainers encouraging and supporting people who open issues to make PRs to solve them.
This document facilitates these other good practices by clarifying what can seem like a mysterious process to those who are unfamiliar with it.

This document is also only intended for community practices, it is **not** suitable for solo projects with one maintainer.

#### Community Standards

Interactions with people in the community must always follow the [community standards](https://julialang.org/community/standards/),
including in pull requests, issues, and discussions.

#### Contributing PRs

*   PRs should match the existing code style present in the file.
*   PRs affecting the public API, including adding new features, must update the public documentation.
*   Comments and (possibly internal) docstrings should make the code accessible.
*   PRs that change code must have appropriate tests.
*   Changes to the code must be made via PR, not pushing to master.

#### Reviewing, Approving, and Merging PRs

*   PRs must have 1 approval before they are merged.
*   PR authors should not approve their own PRs.
*   PRs should pass CI tests before being merged.
*   PRs by people __without__ merge rights must have approval from someone who has merge rights (who will usually then merge the PR).
*   PRs by people __with__ merge rights must have approval from someone else, who may or may not have merge rights (and then may merge their own PR).
*   PRs by people with merge rights should not be merged by people other than the author (just approved).

#### Releases

*   A release should be made as soon as possible after a bugfix PR is merged.
*   Care and consideration should be given as to when to make a breaking release.
*   If the repository is in a state where there are unreleased changes for an extended period of time in preparation for a release, then the version in the Project.toml should be set to the version number of the intended release, with the -DEV suffix.
*   The person who merged the PR should register the new release of the package.

#### Becoming a Collaborator (gaining merge rights)

*   Collaborator merge rights are typically assigned at an Organizational level for all repositories in a GitHub organization, or at a Team level for a subset of repositories.
*   Before becoming a collaborator, it is usual to:
    *   contribute several PRs,
    *   review constructively and kindly several PRs,
    *   contribute meaningfully to several discussions on issues.
*   You may ask to be added as a collaborator.
It is not rude to ask.

---

# ColPrac: Further Guidance

This page offers some further guidance on conventions that can be helpful when collaborating on projects.
This is an expansion on the Collaborative Practices, with more details and extra guidance.
Anything detailed here should be considered less important than the main Collaborative Practices.

## Guidance on contributing PRs

*   You should usually open an issue about a bug or possible improvement before opening a PR with a solution.
*   PRs should do a single thing, so that they are easier to review.
    *   For example, fix one bug, or update compatibility, rather than fixing a bunch of bugs, and updating compatibility, and adding a new feature.
*   PRs should add tests which cover the new or fixed functionality.
*   PRs that move code should not also change code, so that they are easier to review.

    *   If only moving code, review for correctness is not required.

    *   If only changing code, then the diff makes it clear what lines have changed.
*   PRs with large improvements to style should not also change functionality.

    *   This is to avoid making large diffs that are not the focus of the PR.
    *   While it is often helpful to fix a few typos in comments on the way past, it is different from using a regex or formatter on the whole project to fix spacing around operators.

*   PRs introducing breaking changes should make this clear when opening the PR.
*   You should not push commits with commented-out tests.
    *   If pushing a commit for which a test is broken, use the `@test_broken` macro.
    *   Commenting out tests while developing locally is okay, but committing a commented-out test increases the risk of it silently not being run when it should be.
*   You should not squash down commits while review is still ongoing.

    *   Squashing commits prevents the reviewer from seeing what commits have been added since the last review.
*   You should help __review__ your PRs, even though you cannot __approve__ your own PRs.

    *   For instance, start the review process by commenting on why certain bits of the code changed, or highlighting places where you would particularly like reviewer feedback.

## Guidance on reviewing PRs

*   Review comments should be phrased as questions, as it shows you are open to new ideas.

    *   For instance, “Why did you change this to X? Doesn’t that prevent Y?” rather than “You should not have changed this, it will prevent Y”.
*   Small review suggestions, such as typo fixes, should make use of the `suggested change` feature.
    *   This makes it easier and more likely for all the smaller changes to be made.
*   Reviewers should continue acting as reviewers until the PR is merged.

## Guidance on Package Releases

### Incrementing the package version

*   Follow the extension of [SemVer 2.0](https://semver.org) encoded in Julia package manager [Pkg.jl](https://julialang.github.io/Pkg.jl).
*   For a version number X.Y.Z, with Major version X, Minor version Y, Patch version Z:
    *   Post-1.0.0: for breaking changes increment X, for non-breaking new features increment Y, for bug-fixes increment Z.
    *   Pre-1.0.0: for breaking changes increment Y, for non-breaking (feature or bug-fix) increment Z.
*   Introducing deprecations is not breaking; removing deprecations is breaking.
*   There is a cost to making breaking releases - downstream packages have to add support for the new version - so there has to be a bigger benefit to making breaking changes.

### Unreleased Changes and -DEV

Following the Collaborative Practices, when there are unreleased changes in the repository for an extended period of time, the version number in the Project.toml should be suffixed with `-DEV`.
This makes it clear that there are unreleased changes.
Which is useful for many things, including quickly understanding why a bug is still occurring, and working out if a bugfix may need to be backported.

Some more details on the use of `-DEV`.

*   After/during/before the PR making the first change of the release, the version number in the Project.toml should be changed to the intended release number should suffixed `-DEV`.

    *   For instance, if the current version is 0.6.3, then the PR making the breaking change could bump it to 0.7.0-DEV.
    *   Things are more complex if a breaking change is made after the version has been suffixed with `-DEV` for a non-breaking change.
        *   This should be rare, since non-breaking changes should be released as soon as possible.
        *   If it does occur, the following rule applies: _if all version numbers to the right of the digit you would increment are zero, then you do not need to change the version; otherwise, you do._
*   Follow-up PRs can then be made, which do not need to increment the version.
*   Once all the follow-up changes have been made, we can make a PR to drop the `-DEV` suffix and make a new release once this PR is merged.
*   Note that locally, when using `pkg”dev Foo...”` to install particular unreleased versions to an environment, Pkg ignores suffixes to the version number.
Pkg treats `0.7.0-DEV` identically to `0.7.0`.
This means you can update the `[compat]` section of a group of packages and test them together.

### Changing dependency compatibility

*   Generally, changing dependency compatibility should be a non-breaking feature.
    *   i.e. pre-1.0, change the patch version number; post-1.0, change the minor version number.
    *   For instance, adding or removing compatibility with a particular __version__ of a current dependency, which may or may not require internal code changes.
    *   This also applies when adding or removing packages as dependencies.
    *   The new feature in question is the ability to use with a different set of packages.
*   Changing a dependency to resolve a bug is a bug-fix.
    *   i.e. pre/post-1.0 change patch version number.
    *   For instance, if a bug in a downstream dependency is causing a problem in your package, restricting compat to not allow that version would be a bug-fix.
*   Changing compatibility with dependencies **may** be a breaking release, if it breaks the user-facing interface.
    That is to say, if the dependency’s API leaks into your API.
    There are three ways that this can happen:
    *   Reexporting a function that has changed.
    *   Returning an object of a type whose behavior has changed.
    *   Subtyping an object that has changed.


## Guidance on automatically enforcing guidelines

Many of these guidelines can and should be enforced automatically.

- **GitHub:** [Defining the mergeability of pull requests](https://help.github.com/en/github/administering-a-repository/defining-the-mergeability-of-pull-requests)


---

## Definition of Public API

The public API is defined as all exported symbols or symbols marked via the `public` keyword (i.e. the full set of names from `Docs.undocumented_names(module)`), plus any unexported symbols that are explicitly documented as part of the public API, for instance, documented as part of standard usage in the README or hosted documentation. Note that the latter part of the definition is a "deprecation path" because the `public` keyword was only introduced in v1.11: it is expected in the future that the public API will be defined strictly as `Docs.undocumented_names(module)`.

For macros, the public API is the documented set of behaviors and functionality.

## Changes that are considered breaking

* Breaking changes are changes which break functionality in the public API.
* Functions or variables in a package are exported or via the `public` keyword are public API. The full set of public names can be found by `names(Package)`
  * `Docs.undocumented_names(module)` are still considered part of the public API, though it is highly recommended that any such names get documented ASAP.
* Changes which break a documentation example or tutorial are breaking.
* A change to macro syntax is breaking if it breaks documented and tested functionality, or if there is changed functionality does not have an alternative which is supported both before and after the change.

## Changes that are not considered breaking

Everything on this list can, in theory, break users' code. See [XKCD#1172](https://xkcd.com/1172/).
However, we consider changes to these things to be non-breaking from the perspective of package versioning.

*   **Bugs:** We may make backwards incompatible behavior changes if the current implementation is clearly broken, that is, if it contradicts the documentation or if a well-understood behavior is not properly implemented due to a bug.
*   **Internal changes:** Non-public API may be changed or removed.
*   **Exception behavior:**
    *   Exceptions may be replaced with non-error behavior. For instance, we may change a function to compute a result instead of raising an exception, even if that error is documented.
    *   Error message text may change.
    *   Exception types may change unless the exception type for a specific error condition is specified in the documentation.
*   **Floating-point numerical details:** The specific floating-point values may change at any time.
    Users should rely only on approximate accuracy, numerical stability, or statistical properties, not on the specific bits computed.
*   **New exports**: Adding a new export is never considered breaking.
    However, one should consider carefully before exporting a commonly used name that might clash with an existing name (especially, if clashing with `Base`).

*   **New supertypes**:
    *   A new supertype may be added to an existing hierarchy.
    That is, changing `A <: B` to `A <: B <: C` or `A <: C <: B`.
    This includes adding a supertype to something without one, i.e. with supertype `Any`.
    *   A `Union` constant may be replaced by an abstract type that covers all elements of the union.
*   **Changes to the string representation:** The output of `print`/`string` or `show`/`repr` on a type may change at any time.
    Users should not depend on the exact text, but rather on the meaning of the text.
    Changing the string representation often breaks downstream packages tests, because it is hard to write test-cases that depend only on meaning (though unit tests with mocking can be shielded from this kind of breaking).

(This guidance on non-breaking changes is inspired by [https://www.tensorflow.org/guide/versions](https://www.tensorflow.org/guide/versions).)

## Acknowledgements

This contribution guidelines have been adapted from the [SciML ColPrac](https://github.com/SciML/ColPrac) guidelines.
