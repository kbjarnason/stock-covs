#%% md
Predict future stock variances/covariances
By Kristian Bjarnason
#%%
using Pkg, Revise, DataFrames, Impute, CSV, LinearAlgebra, Dates, Statistics, MLJ, MLJBase, MLJModels, MLJLinearModels, Plots, Flux, MLBase, StatsBase

#%%md
Constants
#%%
const WINDOW_SIZE = 1000
const NUM_COVS = 10
const startDate = Date("2008-01-01")

#%%md
Functions
#%%
#sum log difference of every 5th item in a 1-dimensional array/vector
function every_fifth_log_diff_sq(x)
    r = similar(x)
    for i in 1:5
        r[i] = 0.0
    end
    for i in 6:length(x)
        if 1 % 5 == 1
            if x[i] == NaN || x[i-5] == NaN
                r[i] = 0.0
            else
                r[i] = (log(x[i]) - log(x[i-5]))^2
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
Using closing prices, compute the realized variance (RV) for each day using 5-minute log returns.
Note that there are 78 price observations each day.
#%%
RVs = DataFrame(Date = Date[])
append!(RVs[:Date], unique(comps[1][:Date]))

for (x,comp) in enumerate(comps)
    compname_RV = compnames[x] * " RV"
    insert!(RVs, ncol(RVs)+1, 0.0, Symbol(compname_RV))
    for (i,date) in enumerate(allDates)
        RVs[i, Symbol(compname_RV)] = every_fifth_log_diff_sq(groupby(comps[x], :Date)[(Date=date,)][:Close])
    end
end

###TESTING###
# x=1
# compname_RV = compnames[x] * " RV"
# insert!(RVs, ncol(RVs)+1, 0.0, Symbol(compname_RV))
#
# date = allDates[1]
# vec(groupby(comps[1], :Date)[(Date=date,)][:Close])
#
# RVs[Symbol(compname_RV)]
#
# every_fifth_log_diff_sq(groupby(comps[1], :Date)[(Date=date,)][:Close])

#%% md
Compute the realized covariance matrix of the vector of 5-minute returns
#%%
Covs = DataFrame(Date = Date[])
append!(Covs[:Date], unique(comps[1][:Date]))

for (x,comp) in enumerate(comps)
    for y in x:length(comps)
        comp_1 = comps[x]
        comp_2 = comps[y]

        comp1_name = compnames[x]
        comp2_name = compnames[y]

        insert!(Covs, ncol(Covs)+1, 0.0, Symbol(comp1_name * "-" * comp2_name * " Cov"))

        for (i,date) in enumerate(allDates)
            Covs[i, Symbol(comp1_name * "-" * comp2_name * " Cov")] = cov(groupby(comps[x], :Date)[(Date=date,)][:Close],groupby(comps[y], :Date)[(Date=date,)][:Close])
        end
    end
end

###TESTING###
# x=1
# y=2
# comp_1 = comps[x]
# comp_2 = comps[y]
#
# comp1_name = compnames[x]
# comp2_name = compnames[y]
#
# insert!(Covs, ncol(Covs)+1, 0.0, Symbol(comp1_name * "-" * comp2_name * " Cov"))
#
# for (i,date) in enumerate(allDates)
#     Covs[i, Symbol(comp1_name * "-" * comp2_name * " Cov")] = cov(groupby(comps[x], :Date)[(Date=date,)][:Close],groupby(comps[y], :Date)[(Date=date,)][:Close])
# end
#
# Covs

#%%md
Compute the trading volume over the day by summing the 1-minute volumes
#%%
TVs = DataFrame(Date = Date[])
append!(TVs[:Date], unique(comps[1][:Date]))

for (x,comp) in enumerate(comps):
    comp_name = compnames[x]

    insert!(TVs, ncol(TVs)+1, 0.0, Symbol(comp_name * ' TV'))

    for date in allDates
        if str(date) == "NaT"
            continue
        else
            TVs['Date']
        end
    end
end

###TESTING###
x=1
TVs = DataFrame(Date = Date[])
append!(TVs[:Date], unique(comps[1][:Date]))

comp_name = compnames[x]
insert!(TVs, ncol(TVs)+1, 0.0, Symbol(comp_name * ' TV'))

for (i,date) in enumerate(allDates)
    if String(date) == "NaT"
        continue
    else
        TVs[i, Symbol(comp_name * ' TV')] = sum(groupby(comps[x], :Date)[(Date=date,)][:Volume])
    end
end
TVs

#%%md
Construct a database consisting of the RVs and trading volumes of each security starting in January 2008 and running until the end of the sample.
#%%
db = merge(Covs, TVs, on=:Date, how=:outer)

CSV.Write("db.csv",db)

#%%md
Import db
#%%
cd("/Users/kristianbjarnason/Documents/Programming/Data/stocks-cov:var/")
db = CSV.Read("db.csv")

#%%md
Consider four different models to forecast future values of the 4×4 realized covariance matrix. One linear penalized regression (LASSO), one Boosted Tree, one Random Forest and one eXtreme Gradient Boosting machine (XGB).
#%%md
Transformations
#%%
for col in colnames(db)
    db[:col + ' sq'] = db[:col]^2
    db[:col + ' log'] = np.log(db[:col])
    db[:col + ' delta'] = db[:col].diff()
    db[:col + ' delta'].fillna(0, inplace = True)
    db[:col + ' sqrt'] = db[:col]**0.5

db_trans = db

CSV.Write("db_trans.csv", db_trans)

#%%md
Import db_trans
#%%
cd("/Users/kristianbjarnason/Documents/Programming/Data/stocks-cov:var/")
db_trans = CSV.Read("db_trans.csv")

#%%md
Set a rolling window with 1,000 observations to estimate the models and to forecast the next day observation. Re-estimate the models for each new window.
#%%md
Lasso
#%%
@load LassoLarsICRegressor
model_lasso = @pipeline std_lasso(std_model = Standardizer(),
                                  lasso = LassoLarsICRegressor(criterion = "bic")
                                  )
preds_lasso = Vector{}

for i in 1:length(db_trans[:Date]) - WINDOW_SIZE
    preds_lasso_daily = Vector{}

    for j in 1:NUM_COVS:
        X_train, X_test = db_trans[i:WINDOW_SIZE+i,:], db_trans[i+1:WINDOW_SIZE+i+1,:]
        y_train = db_trans[i:WINDOW_SIZE+i, j]

        lasso = machine(model_lasso, X_train, y_train)
        fit!(lasso)

        pred_lasso = MLJBase.predict(lasso, X_test)
        pres_lasso_daily.append(pred_lasso)
    end
    preds_lasso.append(preds_lasso_daily)
end

CSV.Write("preds_lasso.csv", preds_lasso)

#%%md
Boosted Tree
#%%
#TODO implement tuning/CV
@load GradientBoostingRegressor
model_BT = @pipeline std_BT(std_model = Standardizer(),
                                  lasso = GradientBoostingRegressor()
                                  )
preds_BT = Vector{}

for i in 1:length(db_trans[:Date]) - WINDOW_SIZE
    preds_BT_daily = Vector{}

    for j in 1:NUM_COVS:
        X_train, X_test = db_trans[i:WINDOW_SIZE+i,:], db_trans[i+1:WINDOW_SIZE+i+1,:]
        y_train = db_trans[i:WINDOW_SIZE+i, j]

        BT = machine(model_BT, X_train, y_train)
        fit!(BT)

        pred_BT = predict()
        pres_BT_daily.append(pred_BT)
    end
    preds_BT.append(preds_BT_daily)
end

CSV.Write("preds_BT.csv", preds_BT)

#%%md
Random Forest
#%%


#%%md
XGBoost
#%%


#%%md
All Predictions
#%%

#%%md
Plot Predictions
#%%
plot(db_trans[WINDOW_SIZE+1:,:10])
plot(preds_lasso)
plot(preds_BT
plot(preds_RF)
plot(preds_XGB)

#%%md
Checking fit (RMSs, MAEs, R-Squared scores)
#%%md
RMSs
#%%
MLJBase.rms(preds_lasso, db_trans[WINDOW_SIZE+1:,:10])
MLJBase.rms(preds_BT, db_trans[WINDOW_SIZE+1:,:10])
MLJBase.rms(preds_RF, db_trans[WINDOW_SIZE+1:,:10])
MLJBase.rms(preds_XGB, db_trans[WINDOW_SIZE+1:,:10])

#%%md
MAEs
#%%
MLJBase.mae(preds_lasso, db_trans[WINDOW_SIZE+1:,:10])
MLJBase.mae(preds_BT, db_trans[WINDOW_SIZE+1:,:10])
MLJBase.mae(preds_RF, db_trans[WINDOW_SIZE+1:,:10])
MLJBase.mae(preds_XGB, db_trans[WINDOW_SIZE+1:,:10])

#%%md
R-Squared scores
#%%
r2(preds_lasso, db_trans[WINDOW_SIZE+1:,:10])
r2(preds_BT, db_trans[WINDOW_SIZE+1:,:10])
r2(preds_RF, db_trans[WINDOW_SIZE+1:,:10])
r2(preds_XGB, db_trans[WINDOW_SIZE+1:,:10])









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
