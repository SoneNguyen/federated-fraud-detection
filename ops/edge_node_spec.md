# Edge node specification

This document describes the operational logic for FX conversion and timezone
resolution. It was intentionally separated from `contracts/schema.json` — the
schema defines field names and types; this document defines how those fields
are computed.

## FX resolution

| Property     | Value                                          |
|-------------|------------------------------------------------|
| Method      | Static rate table (prototype)                  |
| Production  | ECB daily feed or Open Exchange Rates API      |
| Cache TTL   | 900 seconds (from `schema.json fx_cache_ttl_sec`) |
| Fallback    | Last known rate                                |
| On stale    | Set `stale_fx_flag=1`, proceed with last rate  |
| Rates file  | `data/fx/rates.py`                             |

In the prototype, `FXConverter` uses `STATIC_RATES` hardcoded in
`data/fx/rates.py`. No network call is made. The `stale_fx_flag` is always 0
because the rates never expire from a static dict.

In production, replace `FXConverter.__init__` with an API call and store the
`cache_ts` as the response timestamp. The rest of the pipeline is unchanged.

## Timezone resolution

| Client | Currency | Timezone           |
|--------|----------|--------------------|
| 0      | USD      | US/Eastern         |
| 1      | EUR      | Europe/Berlin      |
| 2      | SGD      | Asia/Singapore     |

In the prototype, each client has a hardcoded IANA timezone string in
`CLIENT_CFG` inside the edge node configuration.

In production, the primary lookup is BIN country → IANA timezone using
`pytz` + `tzdata`. Fallback chain:
1. BIN country → IANA tz
2. Merchant country → IANA tz  
3. UTC

The computed `hour_of_day_local` is an integer 0–23 in the cardholder's
local timezone. It is a passthrough feature (not normalized) because
normalizing it would destroy the cyclical meaning of hours.

## Why this is not in schema.json

`schema.json` is a data contract — it defines what the fields are. This
document defines how they are computed. Mixing operational configuration into
the data contract creates two problems:

1. Schema version bumps become required when FX API URLs change.
2. Engineers reading the schema to understand field types are confused by
   operational details.

Any change to the edge node spec does not require a schema version bump,
provided the output field names and types remain unchanged.