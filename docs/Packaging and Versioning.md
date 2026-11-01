# Packaging and Versioning

This project uses the `src` package layout, setuptools for builds, and
`setuptools-scm` to derive its package version from Git tags. The distribution
name is `homemade-neural-network`; the Python import name is
`homemade_neural_network`.

## Tag a release

Commit the code you want to release before creating the tag. A tag identifies
one commit: later commits do not change what that tag installs. From the
repository root, for the next release after `v0.1.0`:

```powershell
git status
git tag -a v0.1.1 -m "Release v0.1.1"
git push                                # To push code
git push upstream v0.1.1                # To push tags
```

Use a new version number for each release. This repository's GitHub remote is
named `upstream`; use `git remote -v` to check your remote name if it differs.
Check local tags with `git tag --list`, inspect the commit with
`git show v0.1.1`, and check the published tag with
`git ls-remote --tags upstream v0.1.1`. A normal `git push` does not
necessarily push tags.

At the tagged commit, `setuptools-scm` produces version `0.1.1`. Builds from
commits after the tag get development versions until the next release tag.
Do not move an already published release tag; create a new tag for a fix.

## (OPTIONAL) Build a package locally

Building a wheel or source archive is optional when installing directly from
GitHub. To produce distributions locally, use a Python environment with the
build tool installed:

```powershell
python -m pip install build
python -m build
```

The artifacts appear in `dist/`. Their version is derived from the current
Git checkout, so a checkout with new commits or uncommitted changes may not
build as the exact version on an earlier tag. Building locally does not upload
anything to GitHub or PyPI.

## Install a tagged release from GitHub

In the target project's Python environment, install the existing `v0.1.0`
release with:

```powershell
python -m pip install "git+https://github.com/elementalneil/homemade-neural-network.git@v0.1.0"
python -m pip show homemade-neural-network
```

Replace `v0.1.0` with a later pushed tag to install another release. The
target environment needs Git available; pip clones that commit and builds the
package with the dependencies declared in `pyproject.toml`. The tag does not
need to be published to PyPI. `pip show` displays the installed version and
location for the interpreter running `python -m pip`.

Use the package in Python as follows:

```python
from homemade_neural_network import ActivationFunctionValues, NeuralNetwork
```

If an environment already has the package installed, use
`python -m pip install --upgrade "git+https://github.com/elementalneil/homemade-neural-network.git@v0.1.1"`
to switch to a newer tag. Select that same Python interpreter in VS Code if
the editor reports an unresolved import.
