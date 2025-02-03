import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import os

file_path = r'/Users/sreerajmuthaiya.a.l/Downloads/Copy of updated_ai_model_records.xlsx'
if os.path.exists(file_path):
    data = pd.read_excel(file_path)
else:
    data = pd.DataFrame(columns=['Date', 'Day', 'Time', 'Number of Consumers', 'Meal Calorie', 'Calorie Wasted'])

def convert_time(x):
    if isinstance(x, str) and ':' in x:
        return int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    return x 

if not data['Time'].empty:
    data['Time'] = data['Time'].apply(convert_time)

X = data[['Number of Consumers', 'Day', 'Time', 'Meal Calorie']]
y = data['Calorie Wasted']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', max_iter=500, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

def predict_and_store(consumers, day, time, total_intake):
    time_in_minutes = convert_time(time)
    features = scaler.transform([[consumers, day, time_in_minutes, total_intake]])
    predicted_calories = model.predict(features)[0]

    new_record = {
        'Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'Day': day,
        'Time': time,
        'Number of Consumers': consumers,
        'Calorie Wasted': predicted_calories,
        'Meal Calorie': total_intake
    }
    new_data = pd.DataFrame([new_record])
    updated_data = pd.concat([data, new_data], ignore_index=True)

    updated_data['Time'] = updated_data['Time'].apply(convert_time)

    updated_data.to_excel(file_path, index=False)

    return predicted_calories
check = 1
while check:
   check = int(input("Enter 0 to stop:"))
   n_consums = int(input("Enter the number of consumers:"))
   day = int(input("Enter day (0-6):"))
   time = input("Enter time in 24H format (HH:MM):")
   calories = int(input("Enter meal calorie:"))
   prediction = predict_and_store(n_consums,day,time,calories)
   actual_calories = int(input("Enter actual wasted calories:"))
   data = {"Day":[day],"Time":[time],"Number of Consumers":[n_consums],"Meal Calorie":[calories],"Calorie Wasted":[actual_calories]}
   with pd.ExcelWriter(r'/Users/sreerajmuthaiya.a.l/Downloads/Copy of updated_ai_model_records.xlsx', mode="a", engine="openpyxl") as writer:
      data.to_excel(writer, index=False)
   print('Updated data saved with clustering information.')
