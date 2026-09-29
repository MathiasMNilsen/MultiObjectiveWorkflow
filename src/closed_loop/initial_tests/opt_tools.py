import numpy  as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline

# Internal
import sim_tools

__all__ = [
    'LinearProxyModel',
    'PolyProxyModel',
    'SVMProxyModel',
    'RandomForestProxyModel',
    'GPRProxyModel',
    'ProxyModelRBF',
    'ProxyModelRBFEfficient',
    'ObjectiveFunction',
    'EnOpt',
]


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

    def __init__(self, X, Y, degree=2, **kwargs):
        
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
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scalerX.transform(X)
        Y_scaled = self.model.predict(X_scaled)
        return self.scalerY.inverse_transform(Y_scaled).squeeze()
    
    def run_fwd_sim(self, state, member):
        state = np.asarray(state)
        return self.predict(state.reshape(1, -1))


from scipy.interpolate import RBFInterpolator
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class RBFProxyModel:

    def __init__(self, X, Y, kernel=None, **kwargs):
        """
        Radial Basis Function (RBF) based proxy model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training input samples
        Y : array-like, shape (n_samples, n_outputs)
            Training target values
        kernel : str, default='linear'
            Kernel type: 'linear', 'thin_plate_spline', 'cubic', 'quintic', 'gaussian', 'multiquadric', 'inverse_multiquadric'
        """
        # Fit model with polynomial features
        self.scalerX = MinMaxScaler(feature_range=(0, 1))
        self.scalerY = MinMaxScaler(feature_range=(0, 1))

        X = self.scalerX.fit_transform(X)
        Y = self.scalerY.fit_transform(Y)

        # 10% for validation using scikit-learn's train_test_split
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )

        if kernel is None:
            # Choose kernel based on validation error
            kernels = [
                'linear', 
                'thin_plate_spline', 
                'cubic', 
                #'quintic', 
                'gaussian', 
                'multiquadric', 
                'inverse_multiquadric'
            ]
            
            errors = []
            for k in kernels:
                model = RBFInterpolator(
                    y=X_train, 
                    d=Y_train,
                    kernel=k,
                    epsilon=kwargs.get('epsilon', 1.0)
                )
                Y_pred = model(X_val)
                errors.append(mean_squared_error(Y_val, Y_pred))
            
            best_kernel_index = np.argmin(errors)
            best_kernel = kernels[best_kernel_index]
            best_error  = errors[best_kernel_index] 
            
            print(f'Selected RBF kernel: {best_kernel} with validation MSE: {best_error:.6f}')
            self.kernel = best_kernel
        else:
            print(f'Using specified RBF kernel: {kernel}')
            self.kernel = kernel
        
        self.model = RBFInterpolator(
                y=X,
                d=Y,
                kernel=self.kernel,
                epsilon=kwargs.get('epsilon', 1.0)
            )

        # Kwargs
        self.keys  = kwargs.get('keys', None)
        self.index = kwargs.get('index', None)

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scalerX.transform(X)
        Y_pred = self.model(X_scaled)
        return self.scalerY.inverse_transform(Y_pred).squeeze()
    
    def run_fwd_sim(self, state, member):
        state = np.asarray(state)
        return self.predict(state.reshape(1, -1))



KERNELS = [
    'linear', 
    'thin_plate_spline', 
    'cubic', 
    'gaussian', 
    'multiquadric', 
    'inverse_multiquadric',
]

class ProxyModelRBF:

    def __init__(self, X, Y, **kwargs):
        
        # Kwargs
        self.well_names = kwargs.get('well_names', None)
        self.data_order = kwargs.get('data_order', None)
        self.time_index = kwargs.get('time_index', None)

        # Fit model with polynomial features
        self.scalerX = MinMaxScaler(feature_range=(0, 1))
        #self.scalerY = MinMaxScaler(feature_range=(0, 1))
        X = self.scalerX.fit_transform(X)
        #Y = self.scalerY.fit_transform(Y)

        # 10% for validation using scikit-learn's train_test_split
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )

        self.data_order_wells = {}
        # Pick out data for each well
        ############################################################
        Y_train_wells = self.Y_to_Ywell(Y_train)  # Use unscaled Y_train
        Y_val_wells   = self.Y_to_Ywell(Y_val)
        Y_full_wells  = self.Y_to_Ywell(Y)

        # Then scale each well's data separately
        self.scalerY_wells = {}
        for well in self.well_names:
            self.scalerY_wells[well] = MinMaxScaler(feature_range=(0, 1))
            Y_full_wells[well]  = self.scalerY_wells[well].fit_transform(Y_full_wells[well])
            Y_train_wells[well] = self.scalerY_wells[well].transform(Y_train_wells[well])
            Y_val_wells[well]   = self.scalerY_wells[well].transform(Y_val_wells[well]) 
            # ... and update train/val accordingly
        ############################################################

        # Make RBF model for each well
        ############################################################
        self.models = {}
        for well in self.well_names:
            err = []
            for kernel in KERNELS:
                model = RBFInterpolator(
                    y=X_train, 
                    d=Y_train_wells[well],
                    kernel=kernel,
                    epsilon=1.0
                )
                Y_pred = model(X_val)
                err.append(mean_squared_error(Y_val_wells[well], Y_pred))
            
            # Pick the best
            best_kernel_index = np.argmin(err)
            best_kernel = KERNELS[best_kernel_index]
            best_error  = err[best_kernel_index] 
            print(f'Well {well}: Selected RBF kernel: {best_kernel} with validation MSE: {best_error:.6f}')
                
            # Set model on full data
            self.models[well] = RBFInterpolator(
                X,
                Y_full_wells[well],
                kernel=best_kernel,
                epsilon=1.0
            )
        ############################################################            
        
    def Y_to_Ywell(self, Y):
        Yout = {key: [] for key in self.well_names}
        for n in range(Y.shape[0]):
            yn_vec = Y[n].squeeze()
            yn_df = sim_tools.vec_to_dataframe(
                vec=yn_vec,
                datatypes=self.data_order,
                index=self.time_index
            )
            for well in self.well_names:
                yn_well_df = sim_tools.data_well(yn_df.copy(), well)
                yn_well_vec = sim_tools.dataframe_to_vec(yn_well_df)

                if n == 0:
                    self.data_order_wells[well] = yn_well_df.columns.tolist()
                
                # Ensure yn_well_vec is 1D before appending
                Yout[well].append(yn_well_vec)
        
        for well in self.well_names:
            Yout[well] = np.array(Yout[well])
        return Yout
            

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scalerX.transform(X)

        # Predict for each well (scaled predictions)
        Y_pred_wells_scaled = {}
        for well, model in self.models.items():
            Y_pred_wells_scaled[well] = model(X_scaled)
        
        # Inverse transform each well's predictions
        Y_pred_wells = {}
        for well in self.well_names:
            Y_pred_wells[well] = self.scalerY_wells[well].inverse_transform(Y_pred_wells_scaled[well])

        # Re-organize to correct order
        Y_pred = []
        for n in range(X.shape[0]):
            yn_df = {}
            for datakey in self.data_order:
                # Find which well this datakey belongs to by splitting at ':'
                well = datakey.split(':')[1]

                # Get the corresponding prediction
                yn_well_df = sim_tools.vec_to_dataframe(
                    vec=Y_pred_wells[well][n].squeeze(),
                    datatypes=self.data_order_wells[well],
                    index=self.time_index
                )
                yn_df[datakey] = yn_well_df[datakey].values
            
            yn_df = pd.DataFrame(yn_df)
            yn_vec = sim_tools.dataframe_to_vec(yn_df)
            Y_pred.append(yn_vec)
        
        Y_pred = np.array(Y_pred)
        return Y_pred.squeeze()
    
    def run_fwd_sim(self, state, member):
        state = np.asarray(state)
        return self.predict(state.reshape(1, -1))




class WellBasedRBFProxyModel:

    def __init__(self, X, Y, **kwargs):
        
        # Kwargs
        self.well_names = kwargs.get('well_names', None)
        self.data_order = kwargs.get('data_order', None)
        self.time_index = kwargs.get('time_index', None)

        # 10% for validation using scikit-learn's train_test_split
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )

        # Separate output data by well
        Ywells = None


        

        

        
        



class ObjectiveFunction:

    def __init__(self, model: LinearProxyModel, target: dict, windpower: np.ndarray, weight=0.5, econ=None):
        self.model = model
        self.target = target
        self.windpower = windpower
        self.weight = weight
        self.econ = econ

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
        vol = sim_tools.production_volume_and_emissions(res, windpower=self.windpower)
        npv = sim_tools.net_present_value(
            volumes=vol,
            econ=self.econ,
        )

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
        dnpv = (npv - Vhat['npv'])/Vhat['npv']

        doil = (Voil - Vhat['oil'])/Vhat['oil']
        dgas = (Vgas - Vhat['gas'])/Vhat['gas']
        dwi  = (Vwi  - Vhat['wi'])/Vhat['wi']
        dwp  = (Vwp  - Vhat['wp'])/Vhat['wp']

        dvol = np.sqrt(doil**2 + dgas**2 + dwi**2 + dwp**2)
        fval = self.weight*dco2  + (1-self.weight)*dvol
        #fval = self.weight*dco2  + (1-self.weight)*np.sqrt(dnpv**2)

        # Print evaluation (in %)
        if eval:
            print("\n" + "="*60)
            print("OBJECTIVE FUNCTION EVALUATION")
            print("="*60)
            print(f"{'Metric':<25} {'Deviation':>15} {'Unit':>10}")
            print("-"*60)
            print(f"{'Net Present Value':<25} {dnpv*100:>14.2f} {'%':>10}")
            print(f"{'CO₂ Emissions':<25} {dco2*100:>14.2f} {'%':>10}")
            print(f"{'Oil Production':<25} {doil*100:>14.2f} {'%':>10}")
            print(f"{'Gas Production':<25} {dgas*100:>14.2f} {'%':>10}")
            print(f"{'Water Injection':<25} {dwi*100:>14.2f} {'%':>10}")
            print(f"{'Water Production':<25} {dwp*100:>14.2f} {'%':>10}")
            print("="*60 + "\n")

        return fval


class Objective:

    def __init__(self, model: LinearProxyModel, target: dict, windpower: np.ndarray, tol=0.01):
        self.model = model
        self.target = target
        self.windpower = windpower
        self.tol = tol

    def __call__(self, x, *args, **kwargs):
        
        # Get index
        index = args[0]
        datatypes = args[1]

        # Define vectorized prediction function
        def predict_and_evaluate(x_sample):
            y = self.model.predict(x_sample.reshape(1, -1))
            ydf = sim_tools.vec_to_dataframe(
            vec=y.squeeze(), 
            datatypes=datatypes,
            index=index
            )
            return self.emissions(ydf)

        # Vectorize the function
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        vectorized_func = np.vectorize(predict_and_evaluate, signature='(n)->()')
        val = vectorized_func(x)
        
        return val.item() if val.size == 1 else val

    def emissions(self, x):
        # Calculate emissions for target
        vol = sim_tools.production_volume_and_emissions(x, windpower=self.windpower)
        dco2 = (vol['co2'] - self.target['co2'])/self.target['co2']
        return dco2
    
    def output_constraints(self, x, *args, **kwargs):
        # return callable constraints for doil, dgas, dwi, dwp within tol
        
        def doil(x, *args):
            y = self.model.predict(x.reshape(1, -1))
            ydf = sim_tools.vec_to_dataframe(
                vec=y.squeeze(), 
                datatypes=args[1],
                index=args[0]
            )
            vol = sim_tools.production_volume_and_emissions(ydf, windpower=self.windpower)
            doil = (vol['oil'] - self.target['oil'])/self.target['oil']
            return self.tol - abs(doil)

        def dgas(x, *args):
            y = self.model.predict(x.reshape(1, -1))
            ydf = sim_tools.vec_to_dataframe(
                vec=y.squeeze(), 
                datatypes=args[1],
                index=args[0]
            )
            vol = sim_tools.production_volume_and_emissions(ydf, windpower=self.windpower)
            dgas = (vol['gas'] - self.target['gas'])/self.target['gas']
            return self.tol - abs(dgas)

        def dwi(x, *args):
            y = self.model.predict(x.reshape(1, -1))
            ydf = sim_tools.vec_to_dataframe(
                vec=y.squeeze(), 
                datatypes=args[1],
                index=args[0]
            )
            vol = sim_tools.production_volume_and_emissions(ydf, windpower=self.windpower)
            dwi = (vol['wi'] - self.target['wi'])/self.target['wi']
            return self.tol - abs(dwi)
        
        def dwp(x, *args):
            y = self.model.predict(x.reshape(1, -1))
            ydf = sim_tools.vec_to_dataframe(
                vec=y.squeeze(), 
                datatypes=args[1],
                index=args[0]
            )
            vol = sim_tools.production_volume_and_emissions(ydf, windpower=self.windpower)
            dwp = (vol['wp'] - self.target['wp'])/self.target['wp']
            return self.tol - abs(dwp)
    
        return doil, dgas, dwi, dwp

        

class EnOpt:

    def __init__(self, fun: callable, cov: np.ndarray, ne: int=100, bounds: list[tuple] = None, njobs: int = 1):
        self.fun = fun
        self.cov = cov
        self.ne = ne 
        self.bounds = bounds
        self.njobs = njobs

    def gradient(self, x, *args):
        
        # Evaluate function
        self.fx = self.fun(x, *args)

        # Generate ensemble
        self.X = np.random.multivariate_normal(mean=x, cov=self.cov, size=self.ne)
        if not self.bounds is None:
            lb = np.array([b[0] for b in self.bounds])
            ub = np.array([b[1] for b in self.bounds])
            self.X  = np.clip(self.X, a_min=lb, a_max=ub)
        # Evaluate function in parallel using multiprocessing
        with Pool(processes=self.njobs) as pool:
            self.F = np.array(pool.starmap(self.fun, [(self.X[n], *args) for n in range(self.ne)]))

        # Calculate gradient
        grad = np.zeros_like(x)
        for n in range(self.ne):
            grad += (self.F[n] - self.fx) * (self.X[n] - x)
        grad = np.linalg.solve(self.cov, grad)/self.ne

        return grad

    def hessian(self, x, *args):
        
        hess = np.zeros((x.size, x.size))
        for n in range(self.ne):
            dF = self.F[n] - self.fx
            dX = (self.X[n] - x)
            hess += dF * (np.outer(dX, dX) - self.cov)
        hess = hess/self.ne
        hess = np.linalg.inv(self.cov) @ hess @ np.linalg.inv(self.cov)
        return np.diag(np.diag(hess))








