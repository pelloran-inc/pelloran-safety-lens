# Contributing

This package is not accepting external contributions yet (v1): pull requests from outside Pelloran cannot be merged for now. Issues are welcome: bug reports, questions about the paper's measurements, and results you got with the package.

Pelloran offers this package under two licences: the GNU Affero General Public License v3.0 and a commercial licence (`COMMERCIAL.md`). Before contributions open, we will add a contributor licence agreement that lets Pelloran license a contribution under both; this file will then say how to sign it.

To check the package locally:

```
python3 -m unittest discover -s tests
```

A change that touches the measurement must leave `scripts/check_results.py` exact on every pool binary you have.

Contact: opensource@pelloran.com
