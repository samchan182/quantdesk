# quantdesk

Derivatives pricing, execution simulation, and pre-trade risk — written from
scratch, with every headline number reproducible from a seed by a single
command.

There is no QuantLib here, and no pricing library of any kind. The point of the
repository is that the pricing is written, tested against closed form, and
defensible line by line.

## State

Under construction, one milestone at a time. `STATUS.md` is the authoritative
checklist; `CLAUDE.md` is the full build specification.

**M0 complete** — scaffold, determinism harness, result-JSON provenance layer.
No headline numbers yet; the first arrives at M3.

## Layout

```
src/quantdesk/
  rng.py          the single seeded-Generator factory; no global random state
  results.py      result-JSON writer: value + seed + inputs + commit + hardware
  config.py       seeds, market parameters, (from M4) contract terms
  analytics/      closed-form Black-Scholes and barrier references
  mc/             path engine, estimators, Greeks, Brownian bridge
  products/       vanilla, autocallable, snowball
  marketdata/     5-minute bars, intraday volume curve
  execution/      impact model, TWAP/VWAP/participation, shortfall
  risk/           pre-trade checks, latency harness
  lifecycle/      order state machine, reconciliation
bench/            one module per headline number; each writes a result JSON
results/          timestamped result JSONs, committed
tests/            acceptance tests
sql/              schema DDL, applied on first boot of the postgres volume
```

## Running it

```bash
python3 -m pip install -e .        # editable; tests also work without it
python3 -m pytest                  # acceptance suite
python3 bench/determinism_selftest.py
docker compose up -d postgres      # needed from M8
```

## Ground rules

Taken from `CLAUDE.md` and binding on every contributor, human or otherwise:

- No number is hardcoded, estimated, or illustrated. Every figure in a report
  comes from a run on this machine.
- No parameter is tuned to make a number land on an expected value.
- No execution logic reads information that postdates the decision it informs.
- Incomplete modules raise. They do not stub, mock, or return a plausible
  default.
- Every benchmark runs from a registered seed and is reproducible.

`DEFENSE.md` carries one entry per headline number: what it means, the mechanism
behind it, what it is sensitive to, what would make it wrong, and the honest
limitation.
