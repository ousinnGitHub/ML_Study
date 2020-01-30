import csv
import random
import datetime

fn = 'data.csv'

with open(fn,'w') as fp:
    # create csv file
    wr = csv.writer(fp)
    # wirte title
    wr.writerow(['date','salesout'])

    # create data
    startDate = datetime.date(2020,1,1)

    for i in range(365):
        amount = 300 + i * 0.1 + random.randrange(50)
        wr.writerow([str(startDate),amount])
        startDate = startDate + datetime.timedelta(days=1)