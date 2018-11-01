#### Read data and rolling window
input_path_base = 'E:/DATA/PAMAP2_Dataset/PAMAP2_Dataset/Protocol/'
tmp_path_base = 'E:/DATA/PAMAP2_Dataset/tmp/'

window_size = 20  # millisecond 
use_rows = 1000000

#### Process and prepare data
pid = [101, 102, 103, 104, 105, 106, 107, 108, 109]
data_columns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20]
label_columns = [1]
