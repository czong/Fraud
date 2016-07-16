import pymssql
import pandas as pd
import ipdb

# Application data
Risk_server = "PRDRSKEDW01.exclaim-prd.com:5510"
Risk_user = 'RiskLinuxUser_R'
Risk_passwd = 'X6N1!#_=8|]52nE'
conn1 = pymssql.connect(Risk_server,Risk_user,Risk_passwd,"RiskDevDm")
cursor1 = conn1.cursor()
myquery1 = ("""
    SELECT  ApplicationNumber,dec_date,dec_month,fpd,fraud01,reporting_channel   
    from    dbo.vDailyRise7 
    where   SystemCode='PDO' and dec_date >= '2016-05-01' and dec_date <= '2016-06-20' and pmt=1 and reporting_channel='Direct Mail' and reporting_customer_type='new'
""")
cursor1.execute(myquery1)
app_data = pd.DataFrame(cursor1.fetchall())
app_data.columns = [_[0] for _ in cursor1.description]

ipdb.set_trace()

# DWO_live (linking between ApplicationNumber and appID in vendors)
# EDW Server -- contains PDO_Live, DWO_live, etc
EDW_server = "PRDBISEDW04.exclaim-prd.com"
EDW_user = 'ElevateSQLUser_R'
EDW_passwd = 'Loans123!'
conn2 = pymssql.connect(EDW_server,EDW_user,EDW_passwd,"DWO_Live")
cursor2 = conn2.cursor()
myquery2 = ("""
        SELECT  str_application_number,appSeriesId
        FROM    dbo.APPLICATION
""")
cursor2.execute(myquery2)
DWO_live = pd.DataFrame(cursor2.fetchall())
DWO_live.columns = [_[0] for _ in cursor2.description]

ipdb.set_trace()

vendor_path = '/esas/SASDataFiles/Rise_ThirdParty_Data'
ida_path = '%s/idanalytics.sas7bdat'%vendor_path


ida_data = pd.read_sas(ida_path)
ipdb.set_trace()
