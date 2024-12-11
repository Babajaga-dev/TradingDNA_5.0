# Parametri da Implementare

## Sezione Simulator
- `data_file`: Non utilizzato per la selezione del file dati
- `returns_limit`: Non implementato per il limite dei rendimenti
- `element_size_bytes`: Non utilizzato per ottimizzazione memoria
- `min_equity`: Non completamente implementato per il controllo equity minima

## Sezione System
### Environment Check
- `min_total_gb`: Non implementato
- `min_available_gb`: Non implementato
- `command_timeout_sec`: Non implementato per timeout comandi GPU

### Process Management
- `nice_high`: Non implementato
- `nice_low`: Non implementato
- `oom_score`: Implementazione parziale
- `gpu_reserve_pct`: Non implementato
- `ram_reserve_pct`: Non implementato

## Sezione Trading
### Indicators
- `rsi.epsilon`: Non implementato
- `rsi.scale`: Non implementato
- `macd.fast_period`: Implementazione incompleta
- `macd.slow_period`: Implementazione incompleta
- `macd.signal_period`: Implementazione incompleta
- `ema.alpha_multiplier`: Non implementato

### Pattern Gene
- `pattern_window.min`: Non completamente implementato
- `pattern_window.max`: Non completamente implementato
- `confirmation_periods`: Non implementato
- `patterns`: Lista pattern non completamente utilizzata

### Momentum Gene
- `momentum_threshold`: Implementazione parziale
- `trend_strength`: Non implementato
- `overbought_level`: Non completamente utilizzato
- `oversold_level`: Non completamente utilizzato

### Volatility Gene
- `atr_limits.min_size`: Non implementato
- `atr_limits.max_size`: Non implementato

## Sezione Genetic
### Optimizer CUDA Config
- `compute_capability`: Implementazione incompleta
- `optimization_level`: Non implementato
- `allow_tf32`: Non implementato
- `benchmark`: Non implementato
- `deterministic`: Non implementato

### Memory Strategy
- `prealloc_threshold`: Non implementato
- `empty_cache_threshold`: Non implementato
- `force_release_threshold`: Non implementato

### Batch Processing
- `adaptive_batching`: Implementazione parziale
- `prefetch_factor`: Non completamente utilizzato
- `overlap_transfers`: Non completamente implementato

### Memory Management
- `cache_mode`: Non completamente implementato
- `defrag_threshold`: Non implementato
- `periodic_gc`: Non implementato

### Adaptive Mutation
- `plateau_max_factor`: Non implementato
- `plateau_base_factor`: Non implementato
- `improvement_min_factor`: Non implementato
- `improvement_base_factor`: Non implementato
- `fitness_std_multiplier`: Non implementato

### Reproduction
- `crossover_probability`: Non completamente implementato
- `strong_mutation_multiplier`: Non implementato
- `max_attempts_multiplier`: Non implementato

### Diversity
- `top_performer_threshold`: Non implementato
- `performance_bonus_limit`: Non implementato
- `performance_bonus_multiplier`: Non implementato
- `injection_fraction`: Non completamente implementato

### Performance Monitoring
- `metrics_interval`: Non implementato
- `memory_warnings`: Non implementato
- `threshold_utilization`: Non implementato

### Fitness Weights Penalties
- `max_trades.limit`: Implementazione hardcoded
- `max_trades.penalty`: Implementazione hardcoded
- `max_drawdown.limit`: Implementazione hardcoded
- `max_drawdown.penalty`: Implementazione hardcoded
- `min_win_rate.limit`: Implementazione hardcoded
- `min_win_rate.penalty`: Implementazione hardcoded

## Sezione Download
- `batch_size`: Non completamente implementato
- `api.key`: Non validato
- `api.secret`: Non validato

## Priorità di Implementazione

### Priorità ALTA
1. Parametri di sistema e memoria
   - min_total_gb
   - min_available_gb
   - gpu_reserve_pct
   - ram_reserve_pct

2. Parametri CUDA e ottimizzazione
   - compute_capability
   - optimization_level
   - memory strategy parameters

3. Parametri trading core
   - macd parameters
   - rsi parameters
   - pattern recognition

### Priorità MEDIA
1. Parametri di gestione processi
   - nice_high/low
   - oom_score

2. Parametri adaptive mutation
   - plateau factors
   - improvement factors

3. Parametri batch processing
   - adaptive_batching
   - prefetch_factor

### Priorità BASSA
1. Parametri di monitoring
   - metrics_interval
   - memory_warnings

2. Parametri diversity
   - performance bonus
   - injection fraction

## Note Implementative
1. Ogni parametro deve essere implementato con:
   - Validazione input
   - Logging appropriato
   - Gestione errori
   - Test unitari

2. Per i parametri hardcoded:
   - Sostituire con valori da config
   - Aggiungere controlli di validità
   - Documentare range validi

3. Per parametri parzialmente implementati:
   - Completare implementazione
   - Aggiungere controlli mancanti
   - Migliorare logging

4. Per nuove implementazioni:
   - Seguire pattern esistenti
   - Mantenere retrocompatibilità
   - Documentare modifiche
