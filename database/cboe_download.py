import ftplib
import zipfile
import tempfile
import pandas as pd
import sqlalchemy as sql
import sqlalchemy.dialects.postgresql as postgres

url = "ftp.datashop.livevol.com"
ftp = ftplib.FTP(url)
ftp.login(user="nexusinvestmentsct@gmail.com", passwd="")
ftp.cwd("subscriptions/order_000003016/item_000004427/")

# Returns iterator of files in directory
files = ftp.mlsd()
files = [x for x in files]
# files is list of tuples
filename = files[-1][0]
# temp file to store zip file
with tempfile.TemporaryFile() as temp:
    ftp.retrbinary("RETR {0}".format(filename), temp.write)
    with zipfile.ZipFile(temp) as zipfile:
        file = zipfile.namelist()[0]
        # open csv from zip and read into dataframe
        with zipfile.open(file) as csv:
            dataframe = pd.read_csv(csv)
ftp.quit()

newframe = pd.DataFrame()
newframe['date'] = dataframe['quote_date']
newframe['high'] = dataframe['high']
newframe['low'] = dataframe['low']
newframe['open'] = dataframe['open']
newframe['close'] = dataframe['close']
newframe['volume'] = dataframe['trade_volume']
newframe['wap'] = dataframe['vwap']
newframe['symbol'] = dataframe['root']
newframe['expiry'] = dataframe['expiration']
newframe['strike'] = dataframe['strike']
newframe['side'] = dataframe['option_type']
newframe['impvol'] = dataframe['implied_volatility_1545']
newframe['delta'] = dataframe['delta_1545']
newframe['gamma'] = dataframe['gamma_1545']
newframe['theta'] = dataframe['theta_1545']
newframe['vega'] = dataframe['vega_1545']
newframe['rho'] = dataframe['rho_1545']
newframe['openinterest'] = dataframe['open_interest']
newframe.set_index("date", inplace=True)

# Drops rows where impvol is zero as data cleaning measure
newframe = newframe[newframe['impvol'] != 0]

# Write data to temp table then merge
engine = sql.create_engine("postgresql://postgres@localhost/optiondata")
# Explicitly define data types for each column to avoid errors on merge
newframe.to_sql('cboe_temp_table', engine, if_exists="replace", dtype={
    "date": postgres.DATE, "high": postgres.NUMERIC, "low": postgres.NUMERIC, "open": postgres.NUMERIC,
    "close": postgres.NUMERIC, "volume": postgres.INTEGER, "wap": postgres.NUMERIC, "symbol": postgres.VARCHAR,
    "expiry": postgres.DATE, "strike": postgres.NUMERIC, "side": postgres.VARCHAR, "impvol": postgres.NUMERIC,
    "delta": postgres.NUMERIC, "gamma": postgres.NUMERIC, "theta": postgres.NUMERIC, "vega": postgres.NUMERIC,
    "rho": postgres.NUMERIC, "openinterest": postgres.INTEGER})
# Explicitly select columns from temp table so order matches market_data table
engine.execute("INSERT INTO market_data SELECT date, high, low, open, close, volume, wap, symbol, expiry, "
               "strike, side, impvol, delta, gamma, theta, vega, rho, openinterest FROM cboe_temp_table "
               "ON CONFLICT DO NOTHING")
engine.execute("DROP TABLE cboe_temp_table")
