# Load data
df = pd.read_csv("AEP_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
df = df.resample("D").mean().dropna()

# GMM
X = df[['AEP_MW']]
X_scaled = StandardScaler().fit_transform(X)
gmm = GaussianMixture(n_components=2, random_state=42).fit(X_scaled)
print("GMM AIC:", gmm.aic(X_scaled))
print("GMM BIC:", gmm.bic(X_scaled))

# Linear Regression
df['time'] = np.arange(len(df))
X_reg = df[['time']]
y_reg = df['AEP_MW']
reg = LinearRegression().fit(X_reg, y_reg)
resid = y_reg - reg.predict(X_reg)
n = len(y_reg)
sigma2 = np.var(resid, ddof=1)
k = X_reg.shape[1] + 1
logL = -0.5 * n * (np.log(2*np.pi*sigma2) + 1)
AIC = -2*logL + 2*k
BIC = -2*logL + k*np.log(n)
print("Linear Regression AIC:", AIC)
print("Linear Regression BIC:", BIC)
