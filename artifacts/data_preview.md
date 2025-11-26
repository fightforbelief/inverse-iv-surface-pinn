# Data Preview — 2025-11-26 03:03:10

- Source dir: `data/raw/`
- Files found: **7**


## 20230103_zero_curve.parquet
- Path: `data/raw/20230103_zero_curve.parquet`
- **ERROR**: ImportError("Missing optional dependency 'tabulate'.  Use pip or conda to install tabulate.")

## SPY20230103_distr_proj.parquet
- Path: `data/raw/SPY20230103_distr_proj.parquet`
- **ERROR**: ImportError("Missing optional dependency 'tabulate'.  Use pip or conda to install tabulate.")

## SPY20230103_option_price.parquet
- Path: `data/raw/SPY20230103_option_price.parquet`
- **ERROR**: ImportError("Missing optional dependency 'tabulate'.  Use pip or conda to install tabulate.")

## SPY20230103_security_price.parquet
- Path: `data/raw/SPY20230103_security_price.parquet`
- **ERROR**: TypeError("float() argument must be a string or a real number, not 'NAType'")

## SPY20230103_stdoption_price.parquet
- Path: `data/raw/SPY20230103_stdoption_price.parquet`
- **ERROR**: ImportError("Missing optional dependency 'tabulate'.  Use pip or conda to install tabulate.")

## SPY20230103_volatility_surface.parquet
- Path: `data/raw/SPY20230103_volatility_surface.parquet`
- **ERROR**: ImportError("Missing optional dependency 'tabulate'.  Use pip or conda to install tabulate.")

## SPY_securd.parquet
- Path: `data/raw/SPY_securd.parquet`
- **ERROR**: TypeError("float() argument must be a string or a real number, not 'NAType'")


---

## Union Schema (columns seen across files)

- date
- secid
- cp_flag
- days
- delta
- impl_volatility
- exdate
- forward_price
- gamma
- strike_price
- theta
- vega
- am_settlement
- amount
- best_bid
- best_offer
- cfadj
- contract_size
- dispersion
- expiry_indicator
- impl_premium
- impl_strike
- last_date
- open_interest
- optionid
- premium
- rate
- root
- ss_flag
- suffix
- symbol
- symbol_flag
- volume