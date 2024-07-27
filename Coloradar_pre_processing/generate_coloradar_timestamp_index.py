
import os

def find_nearest_index(timestamps, target):
    nearest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target))
    return nearest_idx

def generate_new_files(directory):
    print("start!")
    for root, dirs, files in os.walk(directory):

        # for dir in dirs:
        #     print("dir:", dir)

        cnt = 0
        for dir in dirs:
            cnt = cnt + 1
            print("dir:", dir)
            print("cnt", cnt)
            file_a = os.path.join(root, dir, 'single_chip', 'adc_samples', 'timestamps.txt')
            file_b = os.path.join(root, dir, 'lidar', 'timestamps.txt')

            with open(file_a, 'r') as f1, open(file_b, 'r') as f2:
                timestamps_a = [float(line.strip()) for line in f1]
                timestamps_b = [float(line.strip()) for line in f2]

            a = len(timestamps_a)
            b = len(timestamps_b)
        
            if a <= b:
                print("radar less than lidar")
                smaller_file = timestamps_a
                larger_file = timestamps_b
                smaller_file_len = a
                larger_file_len = b
                new_file_a = os.path.join(root, dir, 'single_chip', 'adc_samples', 'radar_index_sequence.txt')
                new_file_b = os.path.join(root, dir, 'lidar', 'lidar_index_sequence.txt')
            else:
                print("radar more than lidar")
                smaller_file = timestamps_b
                larger_file = timestamps_a
                smaller_file_len = b
                larger_file_len = a
                new_file_b = os.path.join(root, dir, 'single_chip', 'adc_samples', 'radar_index_sequence.txt')
                new_file_a = os.path.join(root, dir, 'lidar', 'lidar_index_sequence.txt')

            print(new_file_a)
            print(new_file_b)

            with open(new_file_a, 'w') as f3, open(new_file_b, 'w') as f4:
                for i in range(smaller_file_len):
                    f3.write(str(i) + '\n')
                    nearest_idx = find_nearest_index(larger_file, smaller_file[i])
                    f4.write(str(nearest_idx) + '\n')
        break


# 使用示例
generate_new_files('./coloradar_raw_unzipped')
