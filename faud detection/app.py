from flask import *
app = Flask(__name__) 
import pickle
import numpy as np
import pandas as pd

with open('frauddetect_pkl', 'rb') as f:
    xgb=pickle.load(f)

scaler=scaler=xgb[2]

def retdict(l):
    dc={}
    i=0
    for var in xgb[1]:
        li=[]
        li.append(l[i])
        dc[var]=li
        i=i+1
    k = pd.DataFrame.from_dict(dc)  
    num_df=k[['months_as_customer', 'policy_deductable', 'umbrella_limit','policy_annual_premium',
           'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
           'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
           'vehicle_claim']]
    scaled_data=scaler.transform(num_df)
    scaled_num_df= pd.DataFrame(data=scaled_data, columns=num_df.columns,index=k.index)
    k.drop(columns=scaled_num_df.columns, inplace=True)
    k=pd.concat([scaled_num_df,k],axis=1)
    return k

@app.route('/') 
def home():  
    return render_template("rp.html",sn='')

@app.route('/pred',methods=['GET','POST']) 
def pred():
    month=int(request.form['month']) #month 1
    deductable=int(request.form['deductable']) #deductable 2
    UL=int(request.form['UL']) #umbrella limit 3
    PAP=float(request.form['PAP']) #  policy_annual_premium	4
    cg=int(request.form['cg']) #capital gains 5
    cl=int(request.form['cl']) #capital loss 6
    ihotd=int(request.form['ihotd']) #incident hour of the day 7
    nohi=int(request.form['nohi']) #number_of_vehicles_involved 8
    binj=int(request.form['binj']) #bodily_injuries 9
    witn=int(request.form['witn']) #witnesses 10
    injclaims=int(request.form['injclaims']) #injury_claim 11
    prclaims=int(request.form['prclaims']) #property_claim 12
    vclaims=int(request.form['vclaims']) #vehicle_claim 13
    pcsl=int(request.form['pcsl']) #policy_csl 14
    insured_sex=int(request.form['insured_sex']) #insured_sex 15
    insured_education_level=int(request.form['insured_education_level']) #insured_education_level 16
    incident_severity=int(request.form['incident_severity']) #incident_severity 17
    property_damage=int(request.form['property_damage']) #property_damage 18
    police_report_available=int(request.form['police_report_available']) #police_report_available 19
    insured_occupation=int(request.form['witn']) 
    insured_relationship=int(request.form['insured_relationship'])
    collision_type=int(request.form['collision_type'])
    incident_type=int(request.form['incident_type'])
    authorities_contacted=int(request.form['authorities_contacted'])
    collision_type=int(request.form['collision_type'])
    predlist=[month,deductable,UL,PAP,cg,cl,ihotd,nohi,binj,witn,injclaims,prclaims,vclaims,pcsl,insured_sex,insured_education_level,incident_severity,property_damage,
    police_report_available]
    io=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ir=[0,0,0,0,0,0]
    ct=[0,0,0]
    it=[0,0,0,0]
    ac=[0,0,0,0,0]
    io[insured_occupation]=1
    ir[insured_relationship]=1
    ct[collision_type]=1
    it[incident_type]=1
    ac[authorities_contacted]=1
    predlist=predlist+io+ir+it+ac+ct
    pred=retdict(predlist)
    p=xgb[0].predict(pred)[0]
    if(p==1):
        mess='Might be a fraud!!! You might need to have a look into it personally'
    else:
        mess="Might not be a fraud!!"
    return render_template("rp.html",sn=mess)

if __name__ =='__main__':  
    app.run(debug = True)