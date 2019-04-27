import pandas as pd
import sqlalchemy as sql
import numpy as np
import pyflux as pf
from scipy.optimize import least_squares
from collections import namedtuple


class HorizontalSkew:
    def __init__(self, symbol):
        """
        :param symbol: string containing OPTION ticker not index ticker. ie SPXW not SPX
        """
        self.engine = sql.create_engine("postgresql://postgres@localhost/optiondata")
        self.symbol = symbol
        self.fitted_data = self._skew_model_calc()
        self.recent_data = self._skew_recent_data()

    def _read_sql_query(self, sql_string):
        """
        Returns pandas object using read_sql with given sql command
        :param sql_string: string containing desired SQL command to be run
        :return: pandas dataframe
        """
        return pd.read_sql(sql_string, self.engine)

    @staticmethod
    def _fit_function(x, t, y):
        """
        Function to calculate residuals for linear regression. Model is x + xt + x * sqrt(t)
        :param x: model coefficients
        :param t: x data points
        :param y: variable that can be supplied to calculate residuals
        :return: float
        """
        return (x[0] + (x[1] * t) + (x[2] * np.sqrt(t))) - y

    def _fit_model(self, x_data, y_data):
        """
        Uses scipy least squares function to calculate coefficients for linear model. loss function is adjusted based
        on self.symbol
        :param x_data: x data points given as numpy array
        :param y_data: y data points given as numpy array
        :return: scipy OptimizeResult object. x attribute is an array of the optimized coefficients
        """
        # Starting guess of coefficients. Just 1s
        start = np.ones(3)
        if self.symbol == "RUTW":
            return least_squares(self._fit_function, start, loss="soft_l1", f_scale=0.2, args=(x_data, y_data))
        elif self.symbol == "SPXW":
            return least_squares(self._fit_function, start, loss="soft_l1", f_scale=1, args=(x_data, y_data))
        elif self.symbol == "NDX":
            return least_squares(self._fit_function, start, loss="soft_l1", f_scale=0.15, args=(x_data, y_data))
        else:
            raise ValueError("Invalid symbol given")

    def _most_recent_date(self):
        """
        Selects most recent date in table
        :return: datetime object
        """
        sql_result = self.engine.execute("SELECT DISTINCT date FROM horizontal_skew ORDER BY date DESC LIMIT 1")
        return sql_result.fetchone()[0]

    @staticmethod
    def _create_namedtuple(x, y):
        """
        Function to create namedtuple with x and y attributes
        :param x: desired x data
        :param y: desired y data
        :return: namedtuple
        """
        Points = namedtuple("Points", ["x", "y"])
        return Points(x=x, y=y)

    def _skew_model_calc(self):
        """
        Calculates linear model fit and model values for a range of x's
        :return: namedtuple. x attribute is x data points in numpy array, y attribute is y data points in numpy array
        """
        data = self._read_sql_query("SELECT * FROM horizontal_skew WHERE symbol='{0}'".format(self.symbol))
        dte_array = data['dte'].as_matrix()
        vol_array = data['vol_factor'].as_matrix()

        model = self._fit_model(dte_array, vol_array)

        # Calculating curve for given x values with fitted model
        dte_range = np.arange(350)
        # model.x is fitted coefficients. Zero is given for y to ignore any residuals calculation.
        fitted_function = self._fit_function(model.x, dte_range, 0)

        # Creating namedtuple with x and y attributes
        fitted_data = self._create_namedtuple(dte_range, fitted_function)

        return fitted_data

    def _skew_recent_data(self):
        """
        Gets recent skew data from database
        :return: namedtuple. x attribute is x data points in numpy array, y attribute is y data points in numpy array
        """
        # Selects skew data for most recent day in the database
        recent_date = self._most_recent_date()
        recent_data = self._read_sql_query(
            "SELECT * FROM horizontal_skew WHERE date='{0}' AND symbol='{1}'".format(recent_date, self.symbol))

        # Creating namedtuple with x and y attributes
        recent_data = self._create_namedtuple(recent_data['dte'].as_matrix(),
                                              recent_data['vol_factor'].as_matrix())

        return recent_data


class VolCone:
    def __init__(self, symbol):
        """
        :param symbol: Symbol desired OPTION symbol not index symbol. ie SPXW not SPX
        """
        self.engine = sql.create_engine("postgresql://postgres@localhost/optiondata")
        self.symbol = symbol
        self.vol_cone = self._vol_cone()
        self.current_skew = self._current_horizontal_skew()

    @staticmethod
    def _parse_symbol(symbol):
        """
        Converts option symbol into index symbol. removes W from end. ie SPXW become SPX
        :param symbol: string
        :return: string
        """
        if symbol[-1] == "W":
            return symbol[:-1]
        else:
            return symbol

    def _read_sql_query(self, sql_string):
        """
        Returns pandas object using read_sql with given sql command
        :param sql_string: string containing desired SQL command to be run
        :return: pandas dataframe
        """
        return pd.read_sql(sql_string, self.engine)

    @staticmethod
    def _calc_returns(series):
        """
        Calculates squared log returns
        :param series: pandas series
        :return: pandas series
        """
        return np.square(np.log(series).diff())

    @staticmethod
    def _vol_calc(series):
        """
        Calculate volatility for pandas series or numpy array.
        :param series: series or array of squared return values
        :return: float
        """
        return np.sqrt(np.sum(series) * (1 / (series.size - 1))) * np.sqrt(252)

    def _most_recent_date(self):
        """
        Selects most recent date in table
        :return: datetime object
        """
        sql_result = self.engine.execute("SELECT DISTINCT date FROM horizontal_skew ORDER BY date DESC LIMIT 1")
        return sql_result.fetchone()[0]

    def _vol_cone(self):
        """
        Calculates volatility cone
        :return: dataframe with columns "std_max" "std_min" and "std_avg"
        """
        index_symbol = self._parse_symbol(self.symbol)

        data = self._read_sql_query(
            ("SELECT * FROM underlying_data "
             "WHERE symbol='{0}' AND bar_length='1440' AND data_type='TRADES' ORDER BY date").format(index_symbol)
        )
        returns = self._calc_returns(data['close'])

        windows = [x + 2 for x in range(120)]
        results_dict = {}

        for window in windows:
            std = returns.rolling(window=window).apply(self._vol_calc)
            std_max = std.max()
            std_min = std.min()
            std_avg = std.mean()
            result = pd.Series({"std_max": std_max, "std_min": std_min, "std_avg": std_avg})
            results_dict[window] = result

        return pd.DataFrame.from_dict(results_dict, orient="index")

    def _current_horizontal_skew(self):
        """
        Gets current horizontal skew. ATM volatility for each expiry
        :return: namedtuple with x being dte and y being corresponding atm_vol each a numpy array
        """
        recent_date = self._most_recent_date()
        data = self._read_sql_query(
            ("SELECT * FROM horizontal_skew WHERE symbol='{0}' "
             "AND date='{1}'").format(self.symbol, recent_date)
        )

        Points = namedtuple("Points", ["x", "y"])
        recent_data = Points(x=data['dte'].as_matrix(), y=data['atm_vol'].as_matrix())

        return recent_data


class HistoricalVolatility:
    def __init__(self, symbol):
        """
        :param symbol: INDEX symbol not index symbol ie SPX not SPXW
        """
        self.symbol = symbol
        self.engine = sql.create_engine("postgresql://postgres@localhost/optiondata")
        self.hv10 = self._hv10()
        self.hv30 = self._hv30()
        self.hv60 = self._hv60()

    def _read_sql_query(self, sql_string):
        """
        Returns pandas object using read_sql with given sql command
        :param sql_string: string containing desired SQL command to be run
        :return: pandas dataframe
        """
        return pd.read_sql(sql_string, self.engine)

    @staticmethod
    def _vol_calc(series):
        """
        Calculate volatility for pandas series or numpy array.
        :param series: series or array of squared return values
        :return: float
        """
        return np.sqrt(np.sum(series) * (1 / (series.size - 1))) * np.sqrt(252)

    @staticmethod
    def _calc_returns(series):
        """
        Calculates squared log returns
        :param series: pandas series
        :return: pandas series
        """
        return np.square(np.log(series).diff())

    @staticmethod
    def _create_namedtuple(x, y):
        """
        Function to create namedtuple with x and y attributes
        :param x: desired x data
        :param y: desired y data
        :return: namedtuple
        """
        Points = namedtuple("Points", ["x", "y"])
        return Points(x=x, y=y)

    def _hv10(self):
        """
        Calculates 10 day rolling volatility
        :return: namedtuple containing numpy arrays
        """
        sql_string = "SELECT * FROM underlying_data WHERE symbol='{0}' " \
                     "AND bar_length=1440 AND data_type='TRADES' ORDER BY date".format(self.symbol)
        data = self._read_sql_query(sql_string)
        returns = self._calc_returns(data['close'])

        roll_10_vol = returns.rolling(window=10).apply(self._vol_calc)

        return self._create_namedtuple(roll_10_vol.index.values, roll_10_vol.values)

    def _hv30(self):
        """
        Calculates 30 day rolling volatility
        :return: namedtuple containing numpy arrays
        """
        sql_string = "SELECT * FROM underlying_data WHERE symbol='{0}' " \
                     "AND bar_length=1440 AND data_type='TRADES' ORDER BY date".format(self.symbol)
        data = self._read_sql_query(sql_string)
        returns = self._calc_returns(data['close'])

        roll_30_vol = returns.rolling(window=30).apply(self._vol_calc)

        return self._create_namedtuple(roll_30_vol.index.values, roll_30_vol.values)

    def _hv60(self):
        """
        Calculates 60 day rolling volatility
        :return: namedtuple containing numpy arrays
        """
        sql_string = "SELECT * FROM underlying_data WHERE symbol='{0}' " \
                     "AND bar_length=1440 AND data_type='TRADES' ORDER BY date".format(self.symbol)
        data = self._read_sql_query(sql_string)
        returns = self._calc_returns(data['close'])

        roll_60_vol = returns.rolling(window=60).apply(self._vol_calc)

        return self._create_namedtuple(roll_60_vol.index.values, roll_60_vol.values)


class VolDiffIndex(HistoricalVolatility):
    def __init__(self, symbol):
        """
        :param symbol: INDEX symbol not option symbol. ie SPX not SPXW
        """
        super().__init__(symbol)
        self.hv30_index = self._calc_index()

    def _calc_index(self):
        """
        Calculates difference between option implied volatility and both 10 and 60 day volatility
        :return: 2 namedtuples containing numpy arrays. first is hv60 index, second is hv10 index
        """
        sql_string = "SELECT * FROM underlying_data WHERE symbol='{0}' " \
                     "AND bar_length=1440 AND data_type='OPTION_IMPLIED_VOLATILITY' ORDER BY date".format(self.symbol)
        data = self._read_sql_query(sql_string)
        data = data['close'].values
        # Resize arrays to allow for subtraction
        resized_data = self.hv30.y[-data.size:]

        hv30diff = data - resized_data

        hv30diff_tuple = self._create_namedtuple(self.hv60.x, hv30diff)

        return hv30diff_tuple


class VolModelPredict:
    def __init__(self, symbol, steps_ahead):
        """
        :param symbol: INDEX symbol not option symbol ie SPX not SPXW
        """
        self.symbol = symbol
        self.engine = sql.create_engine("postgresql://postgres@localhost/optiondata")
        self.model = self._fit_model()
        self.steps_ahead = steps_ahead
        self.vol_prediction = self._vol_prediction()

    def _read_sql_query(self, sql_string):
        """
        Returns pandas object using read_sql with given sql command
        :param sql_string: string containing desired SQL command to be run
        :return: pandas dataframe
        """
        return pd.read_sql(sql_string, self.engine)

    @staticmethod
    def _calc_reg_returns(series):
        """
        Calculates log returns
        :param series: pandas series
        :return: numpy array
        """
        return np.diff(np.log(series.values))

    @staticmethod
    def _create_namedtuple(x, y):
        """
        Function to create namedtuple with x and y attributes
        :param x: desired x data
        :param y: desired y data
        :return: namedtuple
        """
        Points = namedtuple("Points", ["x", "y"])
        return Points(x=x, y=y)

    def _fit_model(self):
        sql_string = "SELECT * FROM underlying_data WHERE symbol='{0}' " \
                     "AND bar_length='1440' AND data_type='TRADES' ORDER BY date".format(self.symbol)
        data = self._read_sql_query(sql_string)
        returns = self._calc_reg_returns(data['close'])

        model = pf.GARCH(returns, p=1, q=1)
        model.fit()

        return model

    def _vol_prediction(self):
        prediction = self.model.predict(h=self.steps_ahead)
        vol_prediction = np.sqrt(prediction.values) * np.sqrt(252)
        x_points = np.arange(vol_prediction.size)

        return self._create_namedtuple(x_points, vol_prediction)
