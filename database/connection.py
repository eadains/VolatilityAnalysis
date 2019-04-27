import queue
import datetime
import pandas as pd
import sqlalchemy as sql
import sqlalchemy.dialects.postgresql as postgres
from threading import Thread
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract


class QueueMessage:
    def __init__(self, type, message):
        self.type = type
        self.message = message


class TestWrapper(EWrapper):
    """
    The wrapper deals with the action coming back from the IB gateway or TWS instance

    We override methods in EWrapper that will get called when this action happens, like currentTime
    """

    def __init__(self):
        self.data_queue = queue.Queue()

    def error(self, id, errorCode, errorString):
        errormsg = "IB error id: {0}, error code: {1}, string: \n{2}".format(id, errorCode, errorString)
        print(errormsg)

    def currentTime(self, time):
        # Overridden method
        message = QueueMessage("time", time)
        self.data_queue.put(message)

    def contractDetails(self, reqId, contractDetails):
        message = QueueMessage("contractdetails", contractDetails)
        self.data_queue.put(message)

    def contractDetailsEnd(self, reqId):
        message = QueueMessage("contractdetailsend", None)
        self.data_queue.put(message)

    def historicalData(self, reqId , bar):
        info_dict = {"date": bar.date, "open": bar.open, "high": bar.high,
                     "low": bar.low, "close": bar.close, "volume": bar.volume}
                     #"wap": bar.WAP}
        message = QueueMessage("historicaldata", info_dict)
        self.data_queue.put(message)

    def historicalDataEnd(self, reqId, start, end):
        message = QueueMessage("historicaldataend", None)
        self.data_queue.put(message)


class TestApp(TestWrapper, EClient):

    def __init__(self, ipaddress, portid, clientid):
        TestWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        self.connect(ipaddress, portid, clientid)

        thread = Thread(target=self.run)
        thread.start()

        self.currentid = 1

    def get_time(self):
        self.reqCurrentTime()
        time = self.data_queue.get()
        print(time.message)

    def get_contract(self, contract):
        self.reqContractDetails(self.currentid, contract)
        self.currentid += 1
        details_list = []
        while True:
            queue_object = self.data_queue.get()
            if queue_object.type == "contractdetails":
                details_list.append(queue_object.message)
            elif queue_object.type == "contractdetailsend":
                return details_list

    def get_history(self, contract, enddate, datedelta, barsize, type):
        self.reqHistoricalData(self.currentid, contract, enddate, datedelta, barsize, type, 1, 1, False, [])
        self.currentid += 1
        data_list = []
        while True:
            try:
                queue_object = self.data_queue.get(timeout=60)
            except queue.Empty:
                return
            if queue_object.type == "historicaldata":
                data_list.append(queue_object.message)
            elif queue_object.type == "historicaldataend":
                return data_list

if __name__ == "__main__":
    app = TestApp("127.0.0.1", 4001, 1)
    app.get_time()

    def get_ticker(ticker, timedelta, barsize, type):
        """
        Gets last months historical data for given ticker. 30 min bars.
        :param ticker: String with ticker for desired data
        :param timedelta: how far back to get data. string format ex "1 Y"
        :param barsize: bar size. string format ex "1 day" "10 mins"
        :param type: type of data to get ex "TRADES" "OPTION_IMPLIED_VOLATILITY"
        :return: Dataframe containing data [date, close, high, low, open, volume, wap, symbol]
        """
        stkcontract = Contract()
        stkcontract.symbol = ticker
        stkcontract.secType = "IND"

        today = datetime.datetime.today().strftime("%Y%m%d %H:%M:%S")
        # Gets contract details. get-contract returns list with contractdetails objects, summary gets Contract object
        contractdetails = app.get_contract(stkcontract)[0].summary
        print(contractdetails)
        # Last month with above Contract object. 10 mins using TRADES
        market_data = app.get_history(contractdetails, today, timedelta, barsize, type)

        # Dataframe formatting
        data = pd.DataFrame(market_data)
        data['symbol'] = contractdetails.symbol
        data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
        if barsize == "1 day":
            data['bar_length'] = 1440
        elif barsize == "10 mins":
            data['bar_length'] = 10
        data['data_type'] = type
        return data

    # Writing data to temp table then merging to main table. Gets both price data and implied volatility data
    ticker_list = ["SPX", "RUT", "NDX"]
    engine = sql.create_engine('postgresql://postgres@localhost/optiondata')
    for ticker in ticker_list:
        data = get_ticker(ticker, "5 Y", "1 day", "TRADES")
        voldata = get_ticker(ticker, "5 Y", "1 day", "OPTION_IMPLIED_VOLATILITY")
        data.to_sql("temp_table", engine, if_exists='append', index=False, dtype={
            "date": postgres.DATE, "close": postgres.NUMERIC, "high": postgres.NUMERIC, "low": postgres.NUMERIC,
            "open": postgres.NUMERIC, "volume": postgres.INTEGER, "symbol": postgres.VARCHAR,
            "bar_length": postgres.INTEGER, "data_type": postgres.VARCHAR})
        voldata.to_sql("temp_table", engine, if_exists='append', index=False, dtype={
            "date": postgres.DATE, "close": postgres.NUMERIC, "high": postgres.NUMERIC, "low": postgres.NUMERIC,
            "open": postgres.NUMERIC, "volume": postgres.INTEGER, "symbol": postgres.VARCHAR,
            "bar_length": postgres.INTEGER, "data_type": postgres.VARCHAR})
    engine.execute("INSERT INTO underlying_data SELECT date, close, high, low, open, volume, symbol, bar_length, "
                   "data_type FROM temp_table ON CONFLICT DO NOTHING")
    engine.execute("DROP TABLE temp_table")
    app.disconnect()
