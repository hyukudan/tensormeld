# Third-party notices and adoption status

TensorMeld reuses maintained open-source components instead of reimplementing mature
infrastructure. A package being public does not by itself authorize copying its code.

| Component | Adoption in 0.2.0a2 | License | Evidence |
|---|---|---|---|
| psutil 7.2.2 | Optional `system` runtime dependency | BSD-3-Clause | Real local Linux probe and adapter tests |
| gguf 0.19.0 | Optional `gguf` dependency and read-only directory adapter | MIT | Interface/source reviewed; mocked contract tests; upstream integration pending here |
| llama.cpp | Pinned native execution candidate, not incorporated yet | MIT at reviewed revision | Source identity recorded; no build/execution claim |
| llama-halo-hybrid | Pinned alternate native candidate, not incorporated yet | MIT at reviewed revision | Root license checked; backend qualification pending |

Sources, exact revisions and adoption states are in [third_party/registry.json](third_party/registry.json).
Unmodified upstream license texts are retained under [third_party/licenses/](third_party/licenses/).
No upstream source tree or binary is vendored in this snapshot. Dependencies installed
by a package manager retain their own notices; a future bundled installer must include
all transitive licenses and an artifact-specific dependency inventory. The optional
dependency graph is NOT yet a complete hash-locked installer environment.

The registry is an engineering provenance record, not an assertion that every transitive
library or every file in a native tree has been audited. Third-party names imply no
endorsement. The licenses above apply to their components, not automatically to
TensorMeld's original code; no public distribution license for that code has been selected.
