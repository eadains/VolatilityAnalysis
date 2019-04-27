import urllib.request as req
import zipfile
import tempfile
import pandas as pd
import sqlalchemy as sql
import sqlalchemy.dialects.postgresql as postgres


def full_download():
    url = 'https://www.quandl.com/api/v3/databases/EOD/data?api_key='
    # Create temp file and write downloaded zip file into it
    with tempfile.TemporaryFile() as temp:
        file = req.urlopen(url)
        temp.write(file.read())
        # Take temp file as zip, get the first file out of it, open that file and create dataframe
        with zipfile.ZipFile(temp) as zip:
            file = zip.namelist()[0]
            with zip.open(file) as csv:
                dataframe = pd.read_csv(csv)
    return dataframe


def partial_download():
    url = 'https://www.quandl.com/api/v3/databases/EOD/download?api_key=&download_type=partial'
    # Create temp file and write downloaded zip file into it
    with tempfile.TemporaryFile() as temp:
        file = req.urlopen(url)
        temp.write(file.read())
        # Take temp file as zip, get the first file out of it, open that file and create dataframe
        with zipfile.ZipFile(temp) as zip:
            file = zip.namelist()[0]
            with zip.open(file) as csv:
                dataframe = pd.read_csv(csv)
                print("Download Success")
    return dataframe

if __name__ == '__main__':
    frame = partial_download()
    frame.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'dividend', 'split', 'adj_open',
                     'adj_high', 'adj_low', 'adj_close', 'adj_volume']
    print(frame)
    engine = sql.create_engine("postgresql://postgres@localhost/optiondata")
    # Explicitly define data types for each column to avoid errors on merge
    frame.to_sql('quandl_temp_table', engine, if_exists='replace', index=False, dtype={
        'ticker': postgres.VARCHAR, 'date': postgres.DATE, 'open': postgres.NUMERIC, 'high': postgres.NUMERIC,
        'low': postgres.NUMERIC, 'close': postgres.NUMERIC, 'volume': postgres.NUMERIC, 'dividend': postgres.NUMERIC,
        'split': postgres.NUMERIC, 'adj_open': postgres.NUMERIC, 'adj_high': postgres.NUMERIC,
        'adj_low': postgres.NUMERIC, 'adj_close': postgres.NUMERIC, 'adj_volume': postgres.NUMERIC
    })
    print("Temp Table Write Success")
    # Explicitly select columns from temp table so order matches market_data table
    engine.execute("INSERT INTO quandl_data SELECT ticker, date, open, high, low, close, volume, dividend, split, "
                   "adj_open, adj_high, adj_low, adj_close, adj_volume FROM quandl_temp_table ON CONFLICT DO UPDATE "
                   "SET adj_open = EXCLUDED.adj_open, adj_high = EXCLUDED.adj_high, adj_low = EXCLUDED.adj_low, "
                   "adj_close = EXCLUDED.adj_close, adj_volume = EXCLUDED.adj_volume")
    engine.execute("DROP TABLE quandl_temp_table")