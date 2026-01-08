import extract_ISCs as isc
import extract_features as feat


folder_path = '/home/andreaprato/Desktop/BSc Project/dataset/raw/e4/'
#folder_path = '/home/andreaprato/Desktop/BSc Project/dataset/debugging/'

isc.calculate_all_ISCs(folder_path)
feat.calculate_all_features(folder_path)

print('Done-----', data_path)
print()
