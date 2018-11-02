#### Read data and rolling window
input_path_base = 'E:/DATA/PAMAP2_Dataset/PAMAP2_Dataset/Protocol/'
tmp_path_base = 'E:/DATA/PAMAP2_Dataset/tmp/'

window_size = 10  # millisecond 
use_rows = 1000000

#### Process and prepare data
pid = [101, 102, 103, 104, 105, 106, 107, 108, 109]
data_columns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20]
sensor_heart, sensor_temprature, sensor_3d_acc_16g = [2], [3], [4, 5, 6]
sensor_3d_acc_6g, sensor_3d_gyroscope, sensor_3d_magnet = [7, 8, 9], [10, 11, 12] , [13, 14, 15]
label_columns = [1]