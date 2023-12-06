import pandas as pd
import numpy as np
import math

# TODO change file path
file_path = 'Cu_22C.csv'

# Read the CSV file into a pandas DataFrame
# TODO change the number of rows to skip
df = pd.read_csv(file_path)
df = df.iloc[2:,:]

# Convert columns to numeric (in case they contain strings)
df[0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
df[1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
df[2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
df[3] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
df[4] = pd.to_numeric(df.iloc[:, 4], errors='coerce')

# Drop any rows with NaN values after conversion
df = df.dropna(axis=1)

# resample the data
df['time']=pd.to_timedelta(df[0],unit="ms")
df = df.resample("0.5ms",on='time').mean()

# TODO specify the column that has time, H, and B
time = df.iloc[:, 0].to_numpy()
H_field_list = df[3].to_numpy()
B_field_list = df[4].to_numpy()

# H_0, B_0 = H_field_list[0], B_field_list[0]

# calculate the area of B-H loop with trapezium rule

area = 0
time_error = (float(time[1])-float(time[0]))/2

H_diff = np.diff(H_field_list)
B_diff = np.diff(B_field_list)

start_idx = (H_field_list < -20000).argmin()
end_idx = (H_field_list[start_idx+10:] < -20000).argmax()+10

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

grad_list = B_diff[start_idx:end_idx]/H_diff[start_idx:end_idx]
# grad_list = moving_average(B_diff[start_idx:end_idx],20)/moving_average(H_diff[start_idx:end_idx],20)
# grad_list = grad_list/(4e-7*math.pi)

# area = np.abs(B_field_list[start_idx:end_idx])*np.abs(H_diff[start_idx:end_idx])
area = B_field_list[start_idx:end_idx]*H_diff[start_idx:end_idx]
total_area = np.abs(np.sum(area))
time_for_1loop = 1/50

'''
area_segment = []
for i in range(len(H_field_list)-1):
    area += ((B_field_list[i]) + (B_field_list[i+1])) * (H_field_list[i] - H_field_list[i+1])/2
    area_segment.append(((B_field_list[i]) + (B_field_list[i+1])) * (H_field_list[i] - H_field_list[i+1])/2)
    if H_field_list[i] - H_field_list[0]<0.5 and i > 1000:
        time_for_1loop = abs(float(time[0])-float(time[i]))
        break


#calculate the range of relative permeability by finding gradient
grad_list = []
edge_of_loop = []
for i in range(1,len(H_field_list)-1):
    if H_field_list[i+1] - H_field_list[i] < 0 and H_field_list[i] - H_field_list[i-1]>0:
        edge_of_loop.append(i)
    elif H_field_list[i+1] - H_field_list[i] < 0 and H_field_list[i] - H_field_list[i-1]>0:
        edge_of_loop.append(i)

#for mild steel and transformer iron
for i in range(edge_of_loop[0],edge_of_loop[1]):
    if H_field_list[i+1] == H_field_list[i] or B_field_list[i+1] == B_field_list[i]:
        continue
    grad_list.append(abs((B_field_list[i+1]) - (B_field_list[i]))/abs((H_field_list[i]) - (H_field_list[i+1]))/(4e-7*math.pi))

'''
mu_min = min(np.abs(grad_list))
mu_max = max(np.abs(grad_list))
# print(time_for_1loop)
power = total_area / time_for_1loop
print(f'total_area: {total_area}, power: {power}, mu_min: {mu_min}, mu_max: {mu_max}')


# TODO change the error in measured quantities
frac_error = np.sqrt(10**(-6) + (0.005/9.796)**2 + (0.0001/1.0235)**2 + 
                        (1/500)**2 + (0.052/2.592)**2 )

print(f'error in power is {frac_error*power}, error in grad is {frac_error*mu_min, frac_error*mu_max}')

df.plot(x='H',y='B')
# plt.show()
'''
H_error = df.iloc[:,3].tolist()  
B_error = df.iloc[:,4].tolist()
area_error_sqr = 0
for i in range(len(area_segment)):
    if H_field_list[i+1] == H_field_list[i] or B_field_list[i+1] == B_field_list[i]:
        continue
    area_error_sqr += ((area_segment[i]**2)*(B_error[i]**2+B_error[i+1]**2)/((B_field_list[i] + B_field_list[i+1])**2)+
                       (area_segment[i]**2)*(H_error[i]**2)/((H_field_list[i])**2))
                       
area_error = math.sqrt(area_error_sqr)
power_error = power*math.sqrt((area_error/area)**2+(time_error/time_for_1loop)**2)
print(area_error)
print(power_error)
'''
