import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from binance.client import Client
import time

class BinanceDataDownloader:
    def __init__(self, api_key: str = '', api_secret: str = ''):
        """
        Inizializza il downloader con le chiavi API di Binance.
        Le chiavi non sono necessarie per i dati storici, ma sono utili per aumentare i rate limits.
        """
        self.client = Client(api_key, api_secret)
        self.BATCH_SIZE = 1000  # Dimensione del batch per richiesta
        
    def download_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str = None,
        output_folder: str = 'data'
    ) -> str:
        """
        Scarica i dati storici da Binance in batch da 1000 elementi.
        
        Args:
            symbol: Simbolo trading (es. 'BTCUSDT')
            interval: Intervallo temporale (es. '1m', '5m', '1h', '1d')
            start_date: Data iniziale in formato 'YYYY-MM-DD'
            end_date: Data finale in formato 'YYYY-MM-DD' (default: data corrente)
            output_folder: Cartella dove salvare i dati
        
        Returns:
            Path del file CSV salvato
        """
        print(f"Downloading {symbol} data from {start_date} to {end_date or 'now'}")
        
        # Converti le date in timestamp
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        if end_date:
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)
        
        # Prepara la cartella di output
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calcola l'intervallo di tempo per ogni batch
        interval_milliseconds = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        
        batch_interval = self.BATCH_SIZE * interval_milliseconds[interval]
        
        # Scarica i dati in batch
        all_klines = []
        current_ts = start_ts
        total_duration = end_ts - start_ts
        processed_duration = 0
        
        while current_ts < end_ts:
            try:
                batch_end = min(current_ts + batch_interval, end_ts)
                
                temp_klines = self.client.get_historical_klines(
                    symbol,
                    interval,
                    start_str=current_ts,
                    end_str=batch_end,
                    limit=self.BATCH_SIZE
                )
                
                if temp_klines:
                    all_klines.extend(temp_klines)
                    processed_duration += batch_end - current_ts
                    progress = (processed_duration / total_duration) * 100
                    
                    print(f"Progress: {progress:.2f}% - Downloaded until {datetime.fromtimestamp(batch_end/1000)}")
                    current_ts = temp_klines[-1][0] + 1  # Timestamp dell'ultima candela + 1ms
                else:
                    print(f"No data available for period starting at {datetime.fromtimestamp(current_ts/1000)}")
                    break
                
                # Rispetta i rate limits
                time.sleep(0.5)  # Aumentato il delay tra le richieste per evitare rate limits
                
            except Exception as e:
                print(f"Error downloading data: {e}")
                time.sleep(5)  # Attendi più a lungo in caso di errore
                continue
        
        if not all_klines:
            raise Exception("No data downloaded!")
        
        # Converti in DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Pulisci e formatta i dati
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rimuovi eventuali duplicati e ordina per timestamp
        df = df.drop_duplicates('timestamp')
        df = df.sort_values('timestamp')
        
        # Verifica la continuità dei dati
        time_diffs = df['timestamp'].diff()
        expected_diff = pd.Timedelta(interval)
        gaps = time_diffs[time_diffs > expected_diff]
        
        if not gaps.empty:
            print("\nWarning: Found gaps in data:")
            for idx in gaps.index:
                gap_start = df['timestamp'][idx-1]
                gap_end = df['timestamp'][idx]
                print(f"Gap from {gap_start} to {gap_end}")
        
        # Salva il file
        output_file = output_path / f"market_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        print(f"Downloaded {len(df)} candlesticks")
        
        return str(output_file)

def main():
    # Configura i parametri
    SYMBOL = "BTCUSDT"  # Simbolo da scaricare
    INTERVAL = "1m"     # Intervallo (1m, 5m, 15m, 1h, 4h, 1d)
    START_DATE = "2024-01-01"  # Data di inizio
    END_DATE = None     # Data di fine (None per scaricare fino ad ora)
    
    # Inizializza il downloader
    downloader = BinanceDataDownloader()
    
    try:
        # Scarica i dati
        output_file = downloader.download_historical_data(
            symbol=SYMBOL,
            interval=INTERVAL,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        print(f"\nDownload completato! I dati sono stati salvati in: {output_file}")
        
    except Exception as e:
        print(f"Errore durante il download: {e}")

if __name__ == "__main__":
    main()