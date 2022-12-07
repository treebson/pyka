import pandas as pd

with open('data/WhatsApp Chat with Janice Quach.txt') as file:
    data = file.readlines()

d = {
    'date': [],
    'time': [],
    'person': [],
    'message': []
}
for i, row in enumerate(data[1:]):
    date_ors = row.split(',')
    date = date_ors[0].rstrip()
    time_ors = ''.join(date_ors[1:]).rstrip().split('-')
    time = time_ors[0].rstrip()
    person_ors = ''.join(time_ors[1:]).rstrip().split(':')
    person = person_ors[0].rstrip()
    if person == '':
        print('----')
        print(data[i-1])
        print(data[i])
        print(data[i+1])
    message = ''.join(person_ors[1:]).rstrip()
    d['date'].append(date)
    d['time'].append(time)
    d['person'].append(person)
    d['message'].append(message)

df = pd.DataFrame(d)
print(df['person'].value_counts())