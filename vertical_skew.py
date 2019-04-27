import pandas as pd
import sqlalchemy as sql
import sqlalchemy.dialects.postgresql as postgres
import sys

symbol_list = ["SPXW", "RUTW", "NDX"]

engine = sql.create_engine("postgresql://postgres@localhost/optiondata")
# Selects dates in market_data that are not in horizontal_skew
dates = pd.read_sql("SELECT DISTINCT date from market_data WHERE NOT "
                    "EXISTS(SELECT 1 FROM vertical_skew WHERE market_data.date=vertical_skew.date);", engine)


def delta_diff(x, delta):
    """
    Function for calculating absolute difference for each option from given delta. used to calculate atm option.
    :param x: pandas series given by dataframe.apply()
    :param delta: delta to calculate difference from
    :return: absolute valaue of (delta - input delta)
    """
    return abs(x['delta'] - delta)

dataframe_list = []

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
            puts = pd.read_sql("SELECT * FROM market_data WHERE symbol='{0}' AND date='{1}' "
                               "AND expiry='{2}' AND side='P'".format(symbol, date, expiry_date),
                               engine, index_col="strike")

            # Find closest strike to 50 delta. comma in args is required for single member tuple
            atm_call_strike = calls.apply(delta_diff, axis=1, args=(0.50,)).idxmin()
            # Get implied volatility for strike
            atm_call_vol = calls.loc[atm_call_strike]['impvol']
            # Divide implied volatility by atm vol
            calls_skew_param = calls['impvol'] / atm_call_vol

            atm_put_strike = puts.apply(delta_diff, axis=1, args=(-0.50,)).idxmin()
            atm_put_vol = puts.loc[atm_put_strike]['impvol']
            puts_skew_param = puts['impvol'] / atm_put_vol

            # Converts series to frame, then adds database columns. reset index from strike to default
            call_frame = calls_skew_param.to_frame()
            call_frame.reset_index(inplace=True)
            call_frame['date'] = date
            call_frame['right'] = "C"
            call_frame['expiry'] = expiry_date
            call_frame['symbol'] = symbol
            # Gets delta for each strike from the above calls object and adds it as another column
            call_frame = call_frame.join(calls.loc[call_frame['strike']]['delta'], on="strike")

            puts_frame = puts_skew_param.to_frame()
            puts_frame.reset_index(inplace=True)
            puts_frame['date'] = date
            puts_frame['right'] = "P"
            puts_frame['expiry'] = expiry_date
            puts_frame['symbol'] = symbol
            puts_frame = puts_frame.join(puts.loc[puts_frame['strike']]['delta'], on="strike")

            combined_frame = pd.concat([call_frame[call_frame['strike'] > atm_call_strike],
                                        puts_frame[puts_frame['strike'] <= atm_put_strike]])

            dataframe_list.append(combined_frame)

if dataframe_list:
    endframe = pd.concat(dataframe_list)
    # Writes data to temp table with explicitly defined datatypes
    endframe.to_sql("temp_table", engine, if_exists="replace", index=False, dtype={
        "date": postgres.DATE, "expiry": postgres.DATE, "impvol": postgres.NUMERIC, "right": postgres.VARCHAR,
        "strike": postgres.NUMERIC, "delta": postgres.NUMERIC, "symbol": postgres.VARCHAR})
    # Explictly select columns from temp table so order matches vertical_skew table
    engine.execute("INSERT INTO vertical_skew SELECT date, expiry, impvol, \"right\", strike, delta, symbol "
                   "FROM temp_table ON CONFLICT DO NOTHING")
    engine.execute("DROP TABLE temp_table")
else:
    print("dataframe_list empty. Likely no new data that needs to be updated.")
    sys.exit(1)
