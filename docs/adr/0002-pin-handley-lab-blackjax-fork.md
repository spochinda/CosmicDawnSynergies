# Pin blackjax to the handley-lab fork for nested sampling

Nested sampling is not in released upstream blackjax; the reference
implementation (adaptive nested slice sampling, Yallup/Kroupa/Handley 2026,
arXiv:2601.23252) lives on the `nested_sampling` branch of
https://github.com/handley-lab/blackjax. We pin that fork at commit `2180e29f`
in requirements.txt rather than waiting for the upstream merge (unknown
timeline) or vendoring the ns module (maintenance burden, silent drift).
Consequence: `pip install blackjax` from PyPI is the wrong package for this
repo. When upstream blackjax ships nested sampling, repoint the requirement and
retire this ADR.
