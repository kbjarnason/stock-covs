#%% md
Predict future stock variances/covariances
By Kristian Bjarnason
#%%
using Pkg, Revise, DataFrames, Impute, CSV, LinearAlgebra, Dates, Statistics, MLJ, MLJBase, MLJModels, MLJLinearModels, Plots, MLBase, StatsBase

#%%md
Constants
#%%
const WINDOW_SIZE = 1000
const NUM_COVS = 10
const startDate = Date("2008-01-01")

#%%md
Functions
#%%
#sum log difference of every 5th item multiplied across 2 1-dimensional array/vectors (for cov/var calcs)
function every_fifth_log_diff_mult(x,y)
    r = similar(x)
    for i in 1:5
        r[i] = 0.0
    end
    for i in 6:length(x)
        if 1 % 5 == 1
            if x[i] == NaN || x[i-5] == NaN || y[i] == NaN || y[i-5] == NaN
                r[i] = 0.0
            else
                r[i] = (log(x[i]) - log(x[i-5])) * (log(y[i]) - log(y[i-5]))
            end
        else
            r[i] = 0.0
        end
        if r[i] == NaN || r[i] < 0.0
            r[i] = 0.0
        end
    end
    return sum(r)
end

function timeRangeInt(startTime::Int, endTime::Int)
    times = Vector{Int}()

    #first hour
    hour = floor(Int, startTime/100)
    for min in startTime%100:59
        append!(times, hour * 100 + min)
    end

    #intermediate hours
    for hour in floor(Int, startTime/100)+1:floor(Int, endTime/100)-1
        for min in 0:59
            append!(times, hour * 100 + min)
        end
    end

    #last hour
    hour = floor(Int, endTime/100)
    for min in 0:endTime%100
        append!(times, hour * 100 + min)
    end
    return times
end

#Credit to chris-b1 https://gist.github.com/chris-b1/50fffa29bf03ab8a6e3925a35b1bab90
ffill(a::AbstractArray) = copy(a)
function ffill(a::AbstractArray{Union{Missing, T}}) where {T}
    res = similar(a, T)
    if ismissing(first(a))
        error("first value in array is missing")
    else
        fillval::T = first(a)
    end

    for i in eachindex(a)
        v = a[i]
        if !ismissing(v)
            fillval = v
        end
        res[i] = fillval
    end
    res
end

#%%md
Import all data and assign names to df
#%%
cd("/Users/kristianbjarnason/Documents/Programming/Data/stocks-cov:var/")

IBM = CSV.read("IBM.txt", copycols=true)
JPM = CSV.read("JPM.txt", copycols=true)
SPY = CSV.read("SPY.txt", copycols=true)
XOM = CSV.read("XOM.txt", copycols=true)

#%%md
Preprocess Data
#%%
comps_og = IBM, JPM, SPY, XOM
comps = [IBM, JPM, SPY, XOM] #how to do??
compnames = "IBM", "JPM", "SPY", "XOM"

for comp in comps
    #drop unnecessary columns
    select!(comp, Not([:High, :Low]))
    #convert date column to Date format
    comp[:Date] = Date.(comp[:Date], Dates.DateFormat("mm/dd/yyyy"))
    #filter out dates before startDate
    filter!(row -> row[:Date] >= startDate, comp)
    #create log return column
    insert!(comp, ncol(comp)+1, 0.0, :LR)

end

allDates = Vector{Date}()
for comp in comps
    for date in unique(comp[:Date])
        if !(date in allDates)
            push!(allDates, date)
        end
    end
end

sort!(allDates)
tradingDayTimes = timeRangeInt(0930, 1559)
times = DataFrame(Date = Date[], Time = Int32[])

for date in allDates
    append!(times[:Date],fill(date, length(tradingDayTimes)))
    append!(times[:Time], tradingDayTimes)
end

for (x,comp) in enumerate(comps)
    comp = join(comp, times, on=[:Date, :Time], kind=:right)
    #forward fill missing values
    colnames = String.(propertynames(comp))
    comp = DataFrame(ffill(Array(comp)))
    rename!(comp, Symbol.(colnames))
    comps[x] = comp
end

#%%md
Using closing prices, compute the realized covariance matrix (RC) for each day using 5-minute log returns.
Note that there are 78 price observations each day.
#%% md
Compute the of the vector of 5-minute returns
#%%
RCs = DataFrame(Date = Date[])
append!(RCs[:Date], unique(comps[1][:Date]))

for (x,comp) in enumerate(comps)
    for y in x:length(comps)
        comp_1 = comps[x]
        comp_2 = comps[y]

        comp1_name = compnames[x]
        comp2_name = compnames[y]

        insert!(RCs, ncol(RCs)+1, 0.0, Symbol(comp1_name * "-" * comp2_name * " RC"))

        for (i,date) in enumerate(allDates)
            RCs[i, Symbol(comp1_name * "-" * comp2_name * " RC")] = every_fifth_log_diff_mult(groupby(comps[x], :Date)[(Date=date,)][:Close], groupby(comps[y], :Date)[(Date=date,)][:Close])
        end
    end
end
#%%md
Compute the trading volume over the day by summing the 1-minute volumes
#%%
TVs = DataFrame(Date = Date[])
append!(TVs[:Date], unique(comps[1][:Date]))

for (x,comp) in enumerate(comps)
    comp_name = compnames[x]

    insert!(TVs, ncol(TVs)+1, 0.0, Symbol(comp_name * " TV"))

    for (i,date) in enumerate(allDates)
        TVs[i, Symbol(comp_name * " TV")] = sum(groupby(comps[x], :Date)[(Date=date,)][:Volume])
    end
end

#%%md
Construct a database consisting of the RVs and trading volumes of each security starting in January 2008 and running until the end of the sample.
#%%
db =  join(RCs, TVs, on=:Date)

CSV.write("db.csv",db)

#%%md
Import db
#%%
cd("/Users/kristianbjarnason/Documents/Programming/Data/stocks-cov:var/")
db = CSV.read("db.csv")

#extract dates
dates = db.Date

#remove dates from df
select!(db, Not(:Date))

#%%md
Consider four different models to forecast future values of the 4×4 realized covariance matrix. One linear penalized regression (LASSO), one Boosted Tree, one Random Forest and one eXtreme Gradient Boosting machine (XGB).
#%%md
Transformations
#%%
db_trans = db

for col in names(db_trans)
    col = string(col)
    if col in ("IBM-IBM RC", "JPM-JPM RC", "SPY-SPY RC", "XOM-XOM RC")
        db_trans[Symbol(col * " log")] = log.(db[Symbol(col)])
        db_trans[Symbol(col * " sqrt")] = db[Symbol(col)].^0.5
    end
    db_trans[Symbol(col * " sq")] = db[Symbol(col)].^2
    db_trans[Symbol(col * " delta")] = vcat(0.0, diff(db[Symbol(col)]))
end

CSV.write("db_trans.csv", db_trans)

#%%md
Import db_trans
#%%
cd("/Users/kristianbjarnason/Documents/Programming/Data/stocks-cov:var/")
db_trans = CSV.read("db_trans.csv")
db = CSV.read("db.csv")
#extract dates
dates = db.Date

ys = [Symbol("IBM-IBM RC log"),
      Symbol("JPM-JPM RC log"),
      Symbol("SPY-SPY RC log"),
      Symbol("XOM-XOM RC log"),
      Symbol("IBM-JPM RC"),
      Symbol("IBM-SPY RC"),
      Symbol("IBM-XOM RC"),
      Symbol("JPM-SPY RC"),
      Symbol("JPM-XOM RC"),
      Symbol("SPY-XOM RC")]

y_training = db_trans[:,ys]

#%%md
Set a rolling window with 1,000 observations to estimate the models and to forecast the next day observation. Re-estimate the models for each new window.
#%%md
Example single model (lasso)
#%%
@load LassoLarsICRegressor
model_lasso = @pipeline std_lasso(std_model = Standardizer(),
                                  lasso = LassoLarsICRegressor(criterion = "bic")
                                  )
preds_lasso = Array{Float64,2}(undef, length(dates) - WINDOW_SIZE, NUM_COVS)
for i in 1:length(dates) - WINDOW_SIZE
    preds_lasso_daily = Vector{}()

    for j in 1:NUM_COVS
        X_train, X_test = db_trans[i:WINDOW_SIZE+i-1,:], DataFrame(db_trans[WINDOW_SIZE+i,:])
        y_train = y_training[i:WINDOW_SIZE+i-1, j]

        lasso = machine(model_lasso, X_train, y_train)
        MLJBase.fit!(lasso)

        pred_lasso = MLJBase.predict(lasso, X_test)
        append!(preds_lasso_daily, pred_lasso)
    end
    preds_lasso[i,:] .= preds_lasso_daily
end

#reconvert log columns
preds_lasso[:,1:4] = exp.(preds_lasso[:,1:4])

CSV.write("preds_lasso.csv", DataFrame(preds_lasso))

#%%md
All Predictions
#%%
@load LassoLarsICRegressor
@load GradientBoostingRegressor
@load RandomForestRegressor pkg=ScikitLearn
@load XGBoostRegressor

model_lasso = @pipeline std_lasso(std_model = Standardizer(),
                                lasso = LassoLarsICRegressor(criterion = "bic"))
model_BT = @pipeline std_BT(std_model = Standardizer(),
                                BT = GradientBoostingRegressor())
model_RF = @pipeline std_RF(std_model = Standardizer(),
                                RF = RandomForestRegressor())
model_XGB = @pipeline std_XGB(std_model = Standardizer(),
                                xgb = XGBoostRegressor())

r1_BT = range(model_BT, :(BT.n_estimators), lower=100, upper=500, scale=:linear);
r2_BT = range(model_BT, :(BT.max_depth), lower=5, upper=15, scale = :linear);
r1_RF = range(model_RF, :(RF.n_estimators), lower=100, upper=500, scale=:linear);
r1_XGB = range(model_XGB, :(xgb.eta), lower=0.005, upper=0.5, scale=:linear);
r2_XGB = range(model_XGB, :(xgb.max_depth), lower=5, upper=15, scale = :linear);

self_tuning_BT_model = TunedModel(model=model_BT,
                                  tuning=Grid(goal=5),
                                  resampling=CV(nfolds=3),
                                  range=[(r1_BT,2), (r2_BT,2)],
                                  measure=rms
                                  )
self_tuning_RF_model = TunedModel(model=model_RF,
                                  tuning=Grid(goal=5),
                                  resampling=CV(nfolds=3),
                                  range=[(r1_RF,3)],
                                  measure=rms
                                  )
self_tuning_xgb_model = TunedModel(model=model_XGB,
                                   tuning=Grid(goal=5),
                                   resampling=CV(nfolds=3),
                                   range=[(r1_XGB,2), (r2_XGB,2)],
                                   measure=rms
                                   )

preds_lasso = Array{Float64,2}(undef, length(dates) - WINDOW_SIZE, NUM_COVS)
preds_BT = Array{Float64,2}(undef, length(dates) - WINDOW_SIZE, NUM_COVS)
preds_RF = Array{Float64,2}(undef, length(dates) - WINDOW_SIZE, NUM_COVS)
preds_XGB = Array{Float64,2}(undef, length(dates) - WINDOW_SIZE, NUM_COVS)

for i in 1:length(dates) - WINDOW_SIZE
    preds_lasso_daily = Vector{}()
    preds_BT_daily = Vector{}()
    preds_RF_daily = Vector{}()
    preds_XGB_daily = Vector{}()

    for j in 1:NUM_COVS
        X_train, X_test = db_trans[i:WINDOW_SIZE+i-1,:], DataFrame(db_trans[WINDOW_SIZE+i,:])
        y_train = y_training[i:WINDOW_SIZE+i-1, j]

        lasso = machine(model_lasso, X_train, y_train)
        self_tuning_BT = machine(self_tuning_BT_model, X_train, y_train)
        self_tuning_RF = machine(self_tuning_RF_model, X_train, y_train)
        self_tuning_XGB = machine(self_tuning_xgb_model, X_train, y_train)

        MLJBase.fit!(lasso)
        MLJBase.fit!(self_tuning_BT)
        MLJBase.fit!(self_tuning_RF)
        MLJBase.fit!(self_tuning_XGB)

        pred_lasso = MLJ.predict(lasso, X_test)
        pred_BT = MLJ.predict(self_tuning_BT, X_test)
        pred_RF = MLJ.predict(self_tuning_RF, X_test)
        pred_XGB = MLJ.predict(self_tuning_XGB, X_test)

        append!(preds_lasso_daily, pred_lasso)
        append!(preds_BT_daily, pred_BT)
        append!(preds_RF_daily, pred_RF)
        append!(preds_XGB_daily, pred_XGB)
    end
    preds_lasso[i,:] .= preds_lasso_daily
    preds_BT[i,:] .= preds_BT_daily
    preds_RF[i,:] .= preds_RF_daily
    preds_XGB[i,:] .= preds_XGB_daily
end

preds_lasso[:,1:4] = exp.(preds_lasso[:,1:4])
preds_BT[:,1:4] = exp.(preds_BT[:,1:4])
preds_RF[:,1:4] = exp.(preds_RF[:,1:4])
preds_XGB[:,1:4] = exp.(preds_XGB[:,1:4])

CSV.write("preds_lasso.csv", DataFrame(preds_lasso))
CSV.write("preds_BT_tuned.csv", DataFrame(preds_BT))
CSV.write("preds_RF_tuned.csv", DataFrame(preds_RF))
CSV.write("preds_XGB_tuned.csv", DataFrame(preds_XGB))

#%%md
Plot Predictions
#%%
plot(Array(db_trans[WINDOW_SIZE+1:nrows(db),1:NUM_COVS]), title="True RCs")
plot(preds_lasso, title="Lasso Predicted RCs")
plot(preds_BT, title="BT Predicted RCs"
plot(preds_RF, title="RF Predicted RCs")
plot(preds_XGB, title="XGB Predicted RCs")

#%%md
Checking fit (RMSs, MAEs, R-Squared scores)
#%%md
RMSs
#%%
MLJBase.rms(vec(preds_lasso), vec(Array(db_trans[WINDOW_SIZE+1:nrows(db),1:NUM_COVS])))
MLJBase.rms(vec(preds_BT), vec(Array(db_trans[WINDOW_SIZE+1:nrows(db),1:NUM_COVS])))
MLJBase.rms(vec(preds_RF), vec(Array(db_trans[WINDOW_SIZE+1:nrows(db),1:NUM_COVS])))
MLJBase.rms(vec(preds_XGB), vec(Array(db_trans[WINDOW_SIZE+1:nrows(db),1:NUM_COVS])))

#%%md
MAEs
#%%
MLJBase.mae(vec(preds_lasso), vec(Array(db_trans[WINDOW_SIZE+1:nrows(db),1:NUM_COVS])))
MLJBase.mae(vec(preds_BT), vec(Array(db_trans[WINDOW_SIZE+1:nrows(db),1:NUM_COVS])))
MLJBase.mae(vec(preds_RF), vec(Array(db_trans[WINDOW_SIZE+1:nrows(db),1:NUM_COVS])))
MLJBase.mae(vec(preds_XGB), vec(Array(db_trans[WINDOW_SIZE+1:nrows(db),1:NUM_COVS])))






#%%md
want the stuff below?? maybe implement later...

#%%md
Construct a global minimum variance portfolio (GMVP) based on the forecasts. Derive the weights of the portfolios as a function of the covariance matrix of the returns.
#%%
GMVPs_ols = Array{}
GMVPs_lasso = Array{}
GMVPs_BT = Array{}
GMVPs_RF = Array{}
GMVPs_xgb = Array{}

ONE4 = Matrix{}([1],[1],[1],[1])
ONE4T = ONE4'

for i in 1:length(dates) - WINDOW_SIZE
    cov_ols = Matrix(

#%%md
Compare the out-of-sample performance of the GMVPs with a equally-weighted portfolio. You should consider four statistics: accumulated return, standard deviation, Sharpe ratio and 5% Value-at-Risk.
#%%md
Import risk-free rate data
#%%
rf = CSV.Read("DGS3MO.csv")

#%%md
Get daily closing prices for each stock/index
#%%

#%%md
Calculate log return for each strategy
#%%


#%%md
Accumulated Returns
#%%


#%%md
Standard Deviations
#%%


#%%md
Sharpe Ratios
#%%


#%%md
5% VaRs
#%%


#%%md
NAVs
#%%
