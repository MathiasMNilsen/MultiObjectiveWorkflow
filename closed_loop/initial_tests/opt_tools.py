import numpy  as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline

# Internal
import sim_tools


class LinearProxyModel:

    def __init__(self, X, Y, **kwargs):
        
        # Data
        self.X = X
        self.Y = Y

        # Fit model
        self.scalerX = MinMaxScaler(feature_range=(0, 1))
        self.scalerY = MinMaxScaler(feature_range=(0, 1))
        self.model = LinearRegression()
        self.model.fit(
            X=self.scalerX.fit_transform(X), 
            y=self.scalerY.fit_transform(Y),
        )

        # Kwargs
        self.keys  = kwargs.get('keys', None)
        self.index = kwargs.get('index', None)

    def predict(self, X):
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scalerX.transform(X)
        Y_scaled = self.model.predict(X_scaled)
        return self.scalerY.inverse_transform(Y_scaled).squeeze()
    
    def run_fwd_sim(self, state, member):
        state = np.asarray(state)
        return self.predict(state.reshape(1, -1))
    


class PolyProxyModel:

    def __init__(self, X, Y, degree=2, alpha=1.0, **kwargs):
        
        # Data
        self.X = X
        self.Y = Y
        self.degree = degree

        # Fit model with polynomial features
        self.scalerX = MinMaxScaler(feature_range=(0, 1))
        self.scalerY = MinMaxScaler(feature_range=(0, 1))
        
        # Create pipeline with polynomial features and Ridge regression
        self.model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=True),
            LinearRegression()
        )
        
        # Fit the model
        self.model.fit(
            X=self.scalerX.fit_transform(X), 
            y=self.scalerY.fit_transform(Y),
        )

        # Kwargs
        self.keys  = kwargs.get('keys', None)
        self.index = kwargs.get('index', None)

    def predict(self, X):
        X_scaled = self.scalerX.transform(X)
        Y_scaled = self.model.predict(X_scaled)
        return self.scalerY.inverse_transform(Y_scaled)
    
    def run_fwd_sim(self, state, member):
        state = np.asarray(state)
        return self.predict(state.reshape(1, -1))



class ObjectiveFunction:

    def __init__(self, model: LinearProxyModel, target: dict, windpower: np.ndarray, weight=0.5):
        self.model = model
        self.target = target
        self.windpower = windpower
        self.weight = weight

    def __call__(self, x, *args, **kwargs):

        # Check shape
        x = np.asarray(x)
        if len(x.shape) == 2:
            ne = x.shape[0]
        else:
            ne = 1
            x  = [x]

        # Get index
        index = args[0]
        datatypes = args[1]

        # Predict
        val = []
        for n in range(ne):
            y = self.model.predict(x[n].reshape(1, -1))
            ydf = sim_tools.vec_to_dataframe(
                vec=y.squeeze(), 
                datatypes=datatypes,
                index=index
            )
            fval = self.func(ydf)
            val.append(fval)

        if ne == 1:
            return val[0]
        else:
            return val

    def func(self, res: pd.DataFrame, eval=False):

        # Calculate emissions for target
        fac = sim_tools.facility_consumption(res, windpower=self.windpower)

        # target emissions and Volumes
        Vhat = self.target

        # Emissions and Volumes
        Mco2 = np.sum(fac['emission_rate [ton/day]'].values) # ton
        Voil = np.sum(res['FOPR'].values)                    # Sm3
        Vgas = np.sum(fac['gas_export [Sm3/day]'].values)    # Sm3
        Vwi  = np.sum(res['FWIR'].values)                    # Sm3
        Vwp  = np.sum(res['FWPR'].values)                    # Sm3

        # objective value
        dco2 = (Mco2 - Vhat['co2'])/Vhat['co2']
        doil = (Voil - Vhat['oil'])/Vhat['oil']
        dgas = (Vgas - Vhat['gas'])/Vhat['gas']
        dwi  = (Vwi  - Vhat['wi'])/Vhat['wi']
        dwp  = (Vwp  - Vhat['wp'])/Vhat['wp']
        fval = self.weight*dco2  + (1-self.weight)*np.sqrt(doil**2 + dgas**2 + dwi**2 + dwp**2)

        # Print evaluation (in %)
        if eval:
            print(f'CO2 deviation: {dco2*100:.2f} %')
            print(f'Oil deviation: {doil*100:.2f} %')
            print(f'Gas deviation: {dgas*100:.2f} %')
            print(f'WI  deviation: {dwi*100:.2f} %')
            print(f'WP  deviation: {dwp*100:.2f} %')

        return fval
        

        
        
        
    


