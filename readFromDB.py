import pymssql
import pandas as pd
import ipdb

#Risk Server -- contains RiskDevDm, DataBank, etc.
Risk_server = "PRDRSKEDW01.exclaim-prd.com:5510"
#Risk_user = 'czong'
#Risk_passwd = '!@Zs871028'
Risk_user = 'RiskLinuxUser_R'
Risk_passwd = 'X6N1!#_=8|]52nE'

#EDW Server -- contains PDO_Live, DWO_live, etc
#EDW_server = "PRDBISEDW04.exclaim-prd.com"
#EDW_user = 'ElevateSQLUser_R'
#EDW_passwd = 'Loans123!'

conn = pymssql.connect(Risk_server,Risk_user,Risk_passwd,"RiskDevDm")
cursor = conn.cursor()
myquery = ("""
    
    select top 10 *
    from dbo.vDailyRise7

""")
cursor.execute(myquery)
ipdb.set_trace()
