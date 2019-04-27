import pandas as pd
import sqlalchemy as sql
import sqlalchemy.dialects.postgresql as postgres
import sys


def delta_diff(x, delta):
    """
    Function for calculating absolute difference for each option from given delta. used to calculate atm option.
    :param x: pandas series given by dataframe.apply()
    :param delta: delta to calculate difference from
    :return: absolute valaue of (delta - input delta)
    """
    return abs(x['delta'] - delta)

symbol_list = ["SPXW", "RUTW", "NDX"]

engine = sql.create_engine("postgresql://postgres@localhost/optiondata")
# Selects dates in market_data that are not in horizontal_skew
dates = pd.read_sql("SELECT DISTINCT date from market_data WHERE NOT "
                    "EXISTS(SELECT 1 FROM horizontal_skew WHERE market_data.date=horizontal_skew.date);", engine)
dict_list = []

for symbol in symbol_list:
    for index, row in dates.iterrows():
        # Gets string format of datetime object YYYY-MM-DD
        date = row['date'].isoformat()
        # list of expirations for historical date
        expirations = pd.read_sql("SELECT DISTINCT expiry FROM market_data WHERE "
                                  "symbol='{0}' AND date='{1}'".format(symbol, date),
                                  engine, index_col="expiry")

        for expiry_index, expiry_row in expirations.iterrows():
            # Get calls and puts for each expiry on historical date. Indexed by strike
            expiry_date = expiry_index.isoformat()
            calls = pd.read_sql("SELECT * FROM market_data WHERE symbol='{0}' AND date='{1}' "
                                "AND expiry='{2}' AND side='C'".format(symbol, date, expiry_date),
                                engine, index_col="strike")

            # Find closest strike to 50 delta. comma in args is required for single member tuple
            atm_strike = calls.apply(delta_diff, axis=1, args=(0.50,)).idxmin()
            # Get implied volatility for strike
            atm_vol = calls.loc[atm_strike]['impvol']

            result = {"date": date, "expiry": expiry_date, "atm_vol": atm_vol, "symbol": symbol}
            dict_list.append(result)

if dict_list:
    endframe = pd.DataFrame(dict_list)
    framelist = []
    # Gets distinct historical data dates
    dates = endframe['date'].unique()
    symbols = endframe['symbol'].unique()
    # Get all expiries for each date and each symbol, sort by expiry, get atm_vol for nearest expiry and then divide all
    # atm_vol's by that nearest one. Also add dte or days to expiration.
    # Appends each frame to list that gets concatenated into the newendframe
    for symbol in symbols:
        for date in dates:
            data = endframe.loc[(endframe['date'] == date) & (endframe['symbol'] == symbol)]
            data.sort_values("expiry", inplace=True)
            # Selects 2nd nearest expiry from date. Avoids weird last trading day vol changes
            near_vol = data['atm_vol'].iloc[1]
            data['vol_factor'] = data['atm_vol'] / near_vol
            # Converts string format to datetime and subtracts, creating timedelta objects
            data['dte'] = pd.to_datetime(data['expiry'], format="%Y-%m-%d") - pd.to_datetime(data['date'], format="%Y-%m-%d")
            # Get just integer of days of timedelta objects
            data['dte'] = data['dte'].dt.days
            # Again, ignoring nearest expiry in favor of 2nd nearest
            framelist.append(data.iloc[1:])

    newendframe = pd.concat(framelist)

    newendframe.to_sql("temp_table", engine, if_exists="replace", index=False, dtype={
        "atm_vol": postgres.NUMERIC, "date": postgres.DATE, "expiry": postgres.DATE, "vol_factor": postgres.NUMERIC,
        "dte": postgres.INTEGER, "symbol": postgres.VARCHAR})
    engine.execute("INSERT INTO horizontal_skew SELECT date, expiry, atm_vol, vol_factor, dte, symbol "
                   "FROM temp_table ON CONFLICT DO NOTHING")
    engine.execute("DROP TABLE temp_table")
else:
    print("dict_list empty. Likely no new data that needs to be updated.")
    sys.exit(1)
