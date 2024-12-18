# TradingDNA 5.0

## Panoramica
TradingDNA 5.0 è un software di trading algoritmico che utilizza algoritmi genetici e deep learning per ottimizzare strategie di trading. Il sistema è progettato per essere altamente performante, supportando sia CPU che GPU (CUDA e Intel Arc).

## Architettura

### Core Components

#### 1. Simulator
- `simulator_processor.py`: Core engine per la simulazione del trading
- Gestisce posizioni multiple in modo vettorizzato
- Supporta position sizing dinamico
- Implementa gestione del rischio (stop loss, take profit, max drawdown)

#### 2. Genetic Optimization
- `genetic_optimizer.py`: Gestione dell'evoluzione genetica
- `genetic_evaluation.py`: Valutazione delle performance
- `genetic_population.py`: Gestione della popolazione
- `genetic_selection.py`: Selezione e riproduzione
- `genetic_adaptation.py`: Adattamento dei parametri

#### 3. Trading Models
- `genes/`: Moduli per diversi tipi di strategie
  - `momentum.py`: Strategie basate sul momentum
  - `pattern.py`: Pattern recognition
  - `volatility.py`: Strategie basate sulla volatilità
  - `indicators.py`: Indicatori tecnici

#### 4. Device Management
- Supporto multi-device:
  - CPU
  - NVIDIA CUDA
  - Intel Arc
- Ottimizzazione memoria e performance

### Configurazione
Il sistema utilizza file YAML per la configurazione:
- `config_cpu.yaml`
- `config_gpu_arc.yaml`
- `config_gpu_cuda.yaml`

Parametri principali:
```yaml
trading:
  position:
    size_pct: 5        # Dimensione posizione in %
    stop_loss_pct: 1.5 # Stop loss in %
    take_profit_pct: 3.0 # Take profit in %

genetic:
  population_size: 80
  generations: 75
  mutation_rate: 0.55
```

## Caratteristiche Principali

### 1. Gestione Posizioni
- Supporto per posizioni multiple concorrenti
- Sizing automatico basato sul capitale
- Gestione del rischio integrata

### 2. Ottimizzazione Genetica
- Popolazione di strategie in evoluzione
- Adattamento automatico dei parametri
- Fitness multi-obiettivo:
  - Profitto
  - Drawdown
  - Sharpe Ratio
  - Win Rate

### 3. Performance
- Operazioni vettorizzate con PyTorch
- Batch processing ottimizzato
- Gestione efficiente della memoria

### 4. Risk Management
- Stop loss dinamici
- Take profit automatici
- Controllo drawdown
- Position sizing adattivo

## Implementazione Tecnica

### Simulazione
```python
class SimulationProcessor:
    def __init__(self, device_manager, config):
        # Parametri trading
        self.position_size_pct = config.get("trading.position.size_pct", 5) / 100
        self.max_positions = math.floor(1 / self.position_size_pct)
        
    def run_simulation(self, prices, entry_conditions, initial_capital):
        # Gestione posizioni multiple
        active_positions = torch.zeros(size, self.max_positions)
        
        # Calcolo PnL e equity
        pnl = calcola_pnl()
        equity = aggiorna_equity()
```

### Ottimizzazione Genetica
```python
class GeneticOptimizer:
    def optimize(self, simulator):
        for generation in range(self.generations):
            # Evoluzione popolazione
            population = evolve_population()
            
            # Valutazione fitness
            fitness = evaluate_fitness()
            
            # Selezione migliori
            best_genes = select_best()
```

## Utilizzo

### Configurazione
1. Selezionare il device target (CPU/GPU)
2. Configurare parametri trading
3. Impostare parametri genetici

### Esecuzione
```bash
python cli.py --config config_gpu_arc.yaml
```

### Monitoraggio
- Log dettagliati delle operazioni
- Report evoluzione genetica
- Grafici performance

## Performance

### Ottimizzazioni
- Batch processing
- Operazioni vettoriali
- Gestione memoria efficiente

### Scalabilità
- Supporto multi-device
- Parallelizzazione
- Gestione carico adattiva

## Conclusioni
TradingDNA 5.0 rappresenta una piattaforma avanzata per il trading algoritmico, combinando:
- Algoritmi genetici per ottimizzazione
- Deep learning per pattern recognition
- Performance computing per simulazioni
- Risk management robusto

Il sistema è progettato per essere:
- Scalabile
- Performante
- Affidabile
- Configurabile
