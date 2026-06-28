# Third-Party Dependency Licenses

This directory contains license texts for third-party dependencies declared by
the benchmark harness and the local documentation website tooling.

The dependency set is intentionally limited to permissive open source licenses
such as MIT, BSD, and Apache-2.0 style licenses.

| Package | Source | Used by | License | License file |
| --- | --- | --- | --- | --- |
| PyYAML | `pyproject.toml`, `website/uv.lock` | Benchmark config parsing and website tooling | MIT | `PyYAML-LICENSE.txt` |
| setuptools | `pyproject.toml` build-system | Python package build backend | MIT | `setuptools-LICENSE.txt` |
| wheel | `pyproject.toml` build-system | Python wheel build support | MIT | `wheel-LICENSE.txt` |
| pytest | `pyproject.toml` optional dev extra | Test runner | MIT | `pytest-LICENSE.txt` |
| exceptiongroup | `pytest` transitive dependency | Test runner compatibility on older Python versions | MIT | `exceptiongroup-LICENSE.txt` |
| iniconfig | `pytest` transitive dependency | Test runner configuration parsing | MIT | `iniconfig-LICENSE.txt` |
| packaging | `pytest` transitive dependency | Test runner version/specifier parsing | Apache-2.0 or BSD | `packaging-LICENSE.txt`, `packaging-LICENSE.APACHE.txt`, `packaging-LICENSE.BSD.txt` |
| pluggy | `pytest` transitive dependency | Test runner plugin system | MIT | `pluggy-LICENSE.txt` |
| zensical | `website/pyproject.toml`, `website/uv.lock` | Documentation website generator | MIT | `zensical-LICENSE.txt` |
| click | `website/uv.lock` | Zensical CLI dependency | BSD-3-Clause | `click-LICENSE.txt` |
| colorama | `website/uv.lock` conditional dependency | Zensical CLI color support on Windows | BSD-3-Clause | `colorama-LICENSE.txt` |
| deepmerge | `website/uv.lock` | Zensical configuration merging | MIT | `deepmerge-LICENSE.txt` |
| Jinja2 | `website/uv.lock` | Zensical template rendering | BSD-3-Clause | `Jinja2-LICENSE.txt` |
| Markdown | `website/uv.lock` | Zensical Markdown rendering | BSD-3-Clause | `Markdown-LICENSE.txt` |
| MarkupSafe | `website/uv.lock` | Jinja2 HTML escaping | BSD-3-Clause | `MarkupSafe-LICENSE.txt` |
| Pygments | `website/uv.lock`, `pytest` transitive dependency | Website syntax highlighting and test output rendering | BSD-2-Clause | `Pygments-LICENSE.txt` |
| pymdown-extensions | `website/uv.lock` | Zensical Markdown extensions | MIT | `pymdown-extensions-LICENSE.txt` |
| tomli | `website/uv.lock`, `pytest` transitive dependency | TOML parsing compatibility | MIT | `tomli-LICENSE.txt` |

When dependency versions change, refresh this directory from the package
metadata shipped in the installed distribution or wheel archive.
