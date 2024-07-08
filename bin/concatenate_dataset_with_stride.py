import pickle
import os

# Define the function to sum values in groups
def somme_par_groupe(liste_tuples, k):
    resultat = []
    n = len(liste_tuples)
    for i in range(0, n, k):
        if i + k <= n:
            somme_y = sum(y for _, y in liste_tuples[i:i+k])
            x = liste_tuples[i][0]
            resultat.append((x, somme_y))
    return resultat

# List of pickle files
pickle_files = [
    '/work/m24046/m24046mrcr/TetQ2500inter_1min_chunk/TetQ2500inter_1min_0_0-500.pkl',
    '/work/m24046/m24046mrcr/TetQ2500inter_1min_chunk/TetQ2500inter_1min_0_500-1000.pkl',
    '/work/m24046/m24046mrcr/TetQ2500inter_1min_chunk/TetQ2500inter_1min_0_1000-1500.pkl',
    '/work/m24046/m24046mrcr/TetQ2500inter_1min_chunk/TetQ2500inter_1min_0_1500-2000.pkl',
    '/work/m24046/m24046mrcr/TetQ2500inter_1min_chunk/TetQ2500inter_1min_0_2000-2100.pkl'
]

# Load and concatenate the lists from each pickle file
combined_list = []
for file in pickle_files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
        combined_list.extend(data)

# Apply the stride function with k=20
k = 20
new_data = somme_par_groupe(combined_list, k)

# Save the new data to a new pickle file
new_pickle_file = '/work/m24046/m24046mrcr/TetQ2500inter_1min_chunk/TetQ2500inter_20min_0_0-2100.pkl'
with open(new_pickle_file, 'wb') as f:
    pickle.dump(new_data, f)

print(f"New pickle file saved as {new_pickle_file}")
