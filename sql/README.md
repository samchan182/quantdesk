# sql/

Schema DDL. `docker-compose.yml` mounts this directory at
`/docker-entrypoint-initdb.d`, so every `.sql` file here is applied, in
filename order, on the **first** boot of an empty `pgdata` volume — and never
again. To re-apply after changing the schema, drop the volume:

```bash
docker compose down -v && docker compose up -d postgres
```

Empty until M8. The tables arrive with the milestones that need them:

| milestone | tables |
|---|---|
| M8 | 5-minute bars, and the intraday volume profile built from them |
| M10 | pre-trade rejection log, written after the fact in `COPY` batches of ten thousand — never on the critical path |
| M11 | order state transitions, fill blotter, daily reconciliation results |

This file also exists so that git tracks the directory at all: git does not
store empty directories, and without it a fresh clone would have no `sql/` for
the compose file to mount.
